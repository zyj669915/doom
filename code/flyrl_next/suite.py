"""可审计、fail-closed 的 suite 调度器；不安装软件、不终止进程。"""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

GIB = 1024 ** 3
RAM_RESERVE = 3 * GIB


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _resolve(base, value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def current_process_memory(output_path=None):
    """采集调用者进程生命周期峰值，不启动训练，也不估计其他配置需求。

    应在 runner/acceptance 进程完成有限 smoke 或实际训练后调用；在 CLI
    单独调用只测 CLI 自身。RSS 包括共享驻留页，commit 是 private committed
    bytes；不包括子进程或显存，不能把短 smoke 峰值当作正式训练安全上界。
    output_path 可把原始测量写入独立 JSON，绝不修改 suite 配置。
    """
    report = {"pid": os.getpid(), "platform": sys.platform, "measured_at_unix": time.time(),
              "scope": "calling_process_lifetime_not_children", "available": False,
              "peak_rss_bytes": None, "rss_bytes": None,
              "peak_commit_bytes": None, "private_commit_bytes": None,
              "vram_bytes": None, "resources": None,
              "note": "仅当前进程生命周期峰值；不涵盖子进程/VRAM，不是正式训练内存上界，不自动生成调度预算。"}
    if sys.platform == "win32":
        from ctypes import wintypes
        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
        try:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.argtypes = []
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            report.update(available=True, peak_rss_bytes=counters.PeakWorkingSetSize,
                          rss_bytes=counters.WorkingSetSize, peak_commit_bytes=counters.PeakPagefileUsage,
                          private_commit_bytes=counters.PrivateUsage, method="GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX)")
        except (OSError, AttributeError) as exc:
            report["error"] = str(exc)
    else:
        report["error"] = "当前 helper 仅支持 Windows；未知内存保持 null"
    if output_path is not None:
        _write(output_path, report)
    return report


def probe_resources():
    """返回当前可用 RAM/commit/VRAM 字节数；无法可靠检测时返回 None。"""
    result = {"ram_available": None, "commit_available": None, "vram_available": None}
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (k, ctypes.c_ulonglong) for k in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result.update(ram_available=memory.avail_phys, commit_available=memory.avail_page)
    elif sys.platform.startswith("linux"):
        try:
            entries = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value = line.split(":", 1)
                entries[name] = int(value.strip().split()[0]) * 1024
            result["ram_available"] = entries["MemAvailable"]
            result["commit_available"] = max(0, entries["CommitLimit"] - entries["Committed_AS"])
        except (OSError, KeyError, ValueError):
            pass
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=True)
        # 多卡必须显式绑定；保守取最小空闲量，不跨卡相加。
        result["vram_available"] = min(int(x.strip()) for x in output.stdout.splitlines()) * 1024 ** 2
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def _requirements(config, supplied):
    if not isinstance(supplied, dict):
        raise ValueError("每个 run 必须显式给出 resources 的 ram_gib/commit_gib/vram_gib")
    result = {}
    for key in ("ram_gib", "commit_gib", "vram_gib"):
        value = supplied.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"资源需求未知或无效: {key}")
        result[key] = int(value * GIB)
    if result["ram_gib"] <= 0 or result["commit_gib"] < result["ram_gib"]:
        raise ValueError("RAM 必须为正，commit 预算不得小于 RAM")
    if str(config.get("device", "cpu")).startswith("cuda") and result["vram_gib"] <= 0:
        raise ValueError("CUDA 必须明确正的 VRAM 预算")
    return result


def _gate(jobs, available):
    requirements = {"ram_available": RAM_RESERVE + sum(j["requirements"]["ram_gib"] for j in jobs),
                    "commit_available": sum(j["requirements"]["commit_gib"] for j in jobs),
                    "vram_available": sum(j["requirements"]["vram_gib"] for j in jobs)}
    errors = []
    for key, required in requirements.items():
        if key == "vram_available" and required == 0:
            continue  # 纯 CPU 明确无需 GPU，而非将未知 GPU 伪装成足够。
        free = available.get(key)
        if not isinstance(free, (int, float)) or not math.isfinite(free):
            errors.append(f"{key} 未知")
        elif free < required:
            errors.append(f"{key} 不足: 需要 {required}，可用 {free}")
    return errors


def _batches(jobs, parallel):
    if not parallel:
        return [[job] for job in jobs]
    pending, batches = list(jobs), []
    while pending:
        batch = [pending.pop(0)]
        other = next((i for i, j in enumerate(pending) if j["kind"] != batch[0]["kind"]), None)
        if other is not None:
            batch.append(pending.pop(other))
        batches.append(batch)
    return batches


def plan(path, parallel=False):
    """suite JSON: {runs:[{id,config,resources}], output?, wrapper?}。

    config 是相对 suite 文件的配置路径；默认 python -m flyrl_next train --config。
    wrapper 可指定新 wrapper 的 .py 路径，绝不调用 v1。
    """
    path = Path(path).resolve()
    spec = _load(path)
    if not isinstance(spec, dict) or not isinstance(spec.get("runs"), list) or not spec["runs"]:
        raise ValueError("suite 必须包含非空 runs 列表")
    output = _resolve(path.parent, spec.get("output", path.stem + "-runs"))
    wrapper = spec.get("wrapper")
    prefix = [sys.executable, str(_resolve(path.parent, wrapper))] if wrapper else [sys.executable, str(Path(__file__).resolve().parents[1] / "flyrl_next_cli.py")]
    jobs, ids = [], set()
    for i, entry in enumerate(spec["runs"]):
        name = entry.get("id", f"run-{i:03d}")
        if not isinstance(name, str) or not name or name in ids or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for ch in name) or name in (".", ".."):
            raise ValueError("run id 必须唯一且只包含安全文件名字符")
        ids.add(name)
        config_path = _resolve(path.parent, entry["config"])
        config = _load(config_path)
        arm = config.get("route", config.get("arm"))
        if arm in ("A1", "A2", "A3", "A7", "real", "rewired", "retinal", "brain"):
            kind = "brain"
        elif arm in ("A4", "A5", "A6", "pixels", "cnn"):
            kind = "cnn"
        else:
            raise ValueError(f"无法确定 run 的脑/CNN 类别: {arm}")
        jobs.append({"id": name, "kind": kind, "config": str(config_path),
                     "requirements": _requirements(config, entry.get("resources")),
                     "command": prefix + ["train", "--config", str(config_path)]})
    available = probe_resources()
    batches = _batches(jobs, parallel)
    gates = [{"ids": [j["id"] for j in batch], "errors": _gate(batch, available)} for batch in batches]
    return {"suite": str(path), "output": str(output), "parallel": bool(parallel), "jobs": jobs,
            "batches": gates, "resources": available, "allowed": not any(g["errors"] for g in gates),
            "ram_reserve_bytes": RAM_RESERVE}


def _execute(job, output):
    folder = output / job["id"]
    folder.mkdir(exist_ok=True)
    training_output = folder / "training"
    training_output.mkdir(exist_ok=True)
    config = _load(job["config"])
    config["output"] = str(training_output)
    effective = folder / "effective-config.json"
    _write(effective, config)
    command = list(job["command"])
    command[-1] = str(effective)
    state = {"id": job["id"], "config": job["config"], "training_output": str(training_output), "state": "starting", "command": command, "started": time.time()}
    _write(folder / "suite-status.json", state)
    try:
        # 不使用 shell；用户配置不能变成 shell 指令。
        with (folder / "stdout.log").open("a", encoding="utf-8") as out, (folder / "stderr.log").open("a", encoding="utf-8") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, cwd=str(Path(__file__).resolve().parents[1]))
            state.update(state="running", pid=process.pid)
            _write(folder / "suite-status.json", state)
            code = process.wait()
        state.update(state="completed" if code == 0 else "failed", returncode=code, finished=time.time())
    except OSError as exc:
        state.update(state="failed", error=str(exc), finished=time.time())
    _write(folder / "suite-status.json", state)
    return state


def run(path, parallel=False):
    """同步调度；默认串行，显式并行也只允许一脑一 CNN，失败后不启动下一批。"""
    prepared = plan(path, parallel=parallel)
    if not prepared["allowed"]:
        raise RuntimeError("资源门禁拒绝: " + json.dumps(prepared["batches"], ensure_ascii=False))
    output = Path(prepared["output"])
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".suite.lock"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("suite 已运行或存在未确认的旧锁；拒绝重复启动") from exc
    states = []
    try:
        os.close(descriptor)
        _write(output / "suite-plan.json", prepared)
        for batch in _batches(prepared["jobs"], parallel):
            if (output / "STOP").exists():
                break
            # 不把 plan 时的快照当成执行时可用资源。
            errors = _gate(batch, probe_resources())
            if errors:
                raise RuntimeError("执行前资源门禁拒绝: " + "; ".join(errors))
            if any((output / job["id"] / "suite-status.json").exists() for job in batch):
                raise RuntimeError("已有执行记录；请使用 resume，不能隐式重训")
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                states.extend(pool.map(lambda j: _execute(j, output), batch))
            if any(s["state"] == "failed" for s in states):
                break
        return {"output": str(output), "runs": states, "stopped": (output / "STOP").exists()}
    finally:
        lock.unlink(missing_ok=True)


def status(path):
    """接受 suite JSON 或 suite/run 目录；仅读取记录，不推断 PID 仍然存活。"""
    path = Path(path).resolve()
    if path.is_file():
        spec = _load(path)
        path = _resolve(path.parent, spec.get("output", path.stem + "-runs"))
    files = [path / "suite-status.json"] if (path / "suite-status.json").exists() else sorted(path.glob("*/suite-status.json"))
    return {"output": str(path), "stop_requested": (path / "STOP").exists(),
            "locked": (path / ".suite.lock").exists(), "runs": [_load(p) for p in files],
            "note": "状态为持久化记录，不代表进程活性；不自动清理旧锁或终止进程。"}


def stop(run_dir):
    """只写 STOP 请求；不发送信号、不 terminate/kill。suite STOP 同时通知已知训练输出。"""
    folder = Path(run_dir).resolve()
    if not folder.is_dir():
        raise ValueError("run_dir 不存在")
    targets = {folder}
    records = list(folder.glob("*/suite-status.json"))
    if (folder / "suite-status.json").exists():
        records.append(folder / "suite-status.json")
    for record in records:
        entry = _load(record)
        if entry.get("training_output"):
            training_output = Path(entry["training_output"])
            targets.update(p.parent for p in training_output.glob("*/config.json"))
        if entry.get("run_dir"):
            targets.add(Path(entry["run_dir"]).resolve())
    for target in targets:
        if target.is_dir():
            (target / "STOP").write_text("cooperative stop requested\n", encoding="utf-8")
    return {"stop_requested": True, "directories": sorted(str(p) for p in targets), "processes_terminated": False}


def resume(run_dir, **kwargs):
    """由 runner 实施 checkpoint/RNG/版本校验；此处不另建恢复算法。"""
    from . import runner
    folder = Path(run_dir).resolve()
    config = runner.load_config(folder / "config.json")
    supplied = config.get("suite_resources", kwargs.pop("resources", None))
    requirements = _requirements(config, supplied)
    errors = _gate([{"requirements": requirements}], probe_resources())
    if errors:
        raise RuntimeError("恢复资源门禁拒绝: " + "; ".join(errors))
    checkpoint = _load(folder / "latest-checkpoint.json")["checkpoint"]
    return runner.train(config, resume=checkpoint, **kwargs)
