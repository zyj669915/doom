"""仅使用标准库的 Windows 资源门禁与监视器；不导入训练库或连接组。"""

from __future__ import annotations

import copy
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any


GIB = 1024 ** 3
MIB = 1024 ** 2
SYSTEM_RESERVE_BYTES = 3 * GIB
ACTIVE_COMMIT_RESERVE_BYTES = 512 * MIB

_MEASUREMENT_LIMITATIONS = [
    "主进程 PeakWorkingSetSize/PeakPagefileUsage 是操作系统记录的整个进程生命周期峰值，包含监视器启动前的用量。",
    "进程树 RSS/private commit 总量及其峰值仅来自离散快照，不是整个进程树的精确生命周期峰值。",
    "children_snapshot_count 是本次快照中递归发现的后代数；累计已见 PID 不能代表全部曾启动的子进程数，PID 也可能复用。",
    "采样间隔内启动并退出的短命后代，以及父进程退出后脱离当前进程树的后代，可能无法观测。",
    "各进程内存读取并非同时发生；RSS 相加可能重复计入共享页，不能当作独占物理内存。",
    "子进程各自的生命周期峰值不能相加当作进程树同时峰值；系统资源门禁与安全停止以全系统可用量为准。",
    "commit_available 使用 GlobalMemoryStatusEx.ullAvailPageFile，受调用进程与系统提交限制约束，并非页面文件剩余磁盘空间。",
]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=1)
def _windows_api() -> tuple[Any, Any]:
    if sys.platform != "win32":
        raise OSError("资源采样仅支持 Windows；未知平台禁止放行。")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCountersEx), wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    return kernel32, psapi


def _win_error(operation: str) -> OSError:
    code = ctypes.get_last_error()
    return OSError(code, f"{operation}失败：{ctypes.FormatError(code).strip()}")


def _system_memory() -> dict[str, int]:
    kernel32, _ = _windows_api()
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise _win_error("读取系统内存")
    return {
        "ram_available_bytes": int(status.ullAvailPhys),
        "commit_available_bytes": int(status.ullAvailPageFile),
        "ram_total_bytes": int(status.ullTotalPhys),
        "commit_limit_bytes": int(status.ullTotalPageFile),
    }


def _snapshot_parent_pids() -> dict[int, int]:
    """只读 Toolhelp 进程快照，不读取或记录任何进程命令行。"""
    kernel32, _ = _windows_api()
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value or snapshot is None:
        raise _win_error("创建进程树快照")
    try:
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            raise _win_error("读取首个进程快照")
        parents: dict[int, int] = {}
        while True:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES 为正常结束。
                    raise _win_error("继续枚举进程快照")
                break
        return parents
    finally:
        kernel32.CloseHandle(snapshot)


def _process_memory(pid: int) -> dict[str, int]:
    kernel32, psapi = _windows_api()
    # 仅申请查询与内存读取权限，绝不申请终止或修改进程权限。
    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
    if not handle:
        raise _win_error(f"打开进程 {pid}")
    try:
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise _win_error(f"读取进程 {pid} 内存")
        return {
            "pid": pid,
            "rss_bytes": int(counters.WorkingSetSize),
            "private_commit_bytes": int(counters.PrivateUsage),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "peak_pagefile_usage_bytes": int(counters.PeakPagefileUsage),
        }
    finally:
        kernel32.CloseHandle(handle)


def _sample_process_tree(root_pid: int) -> dict[str, Any]:
    parents = _snapshot_parent_pids()
    if root_pid not in parents:
        raise OSError("进程快照未包含主进程，无法确认进程树完整性。")
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    pids = [root_pid]
    seen = {root_pid}
    for parent in pids:
        for pid in sorted(children.get(parent, [])):
            if pid not in seen:
                seen.add(pid)
                pids.append(pid)
    # 已退出或无权限的子进程也不能静默忽略，否则会把未知用量当成零。
    processes = [_process_memory(pid) for pid in pids]
    descendants = processes[1:]
    return {
        "root_pid": root_pid,
        "sampled_process_count": len(processes),
        "children_snapshot_count": len(descendants),
        "child_pids": pids[1:],
        "rss_bytes": sum(item["rss_bytes"] for item in processes),
        "private_commit_bytes": sum(item["private_commit_bytes"] for item in processes),
        "children_snapshot_rss_bytes": sum(item["rss_bytes"] for item in descendants),
        "children_snapshot_private_commit_bytes": sum(
            item["private_commit_bytes"] for item in descendants
        ),
        "main_process": processes[0],
        "processes": processes,
    }


def probe_resources() -> dict[str, Any]:
    """返回系统可用内存及当前进程树，所有容量单位为字节；任何未知均 ok=False。"""
    resources: dict[str, Any] = {
        "timestamp_utc": _now(),
        "root_pid": os.getpid(),
        "ok": False,
        "errors": [],
        "ram_available_bytes": None,
        "commit_available_bytes": None,
        "process_tree": None,
        "measurement_limitations": list(_MEASUREMENT_LIMITATIONS),
    }
    try:
        resources.update(_system_memory())
    except Exception as exc:
        resources["errors"].append(f"系统资源采样失败，禁止放行：{exc}")
    try:
        resources["process_tree"] = _sample_process_tree(resources["root_pid"])
    except Exception as exc:
        resources["errors"].append(f"进程树采样失败，禁止放行：{exc}")
    resources["ram_available"] = resources["ram_available_bytes"]
    resources["commit_available"] = resources["commit_available_bytes"]
    resources["ok"] = not resources["errors"]
    return resources


def _budget_bytes(gib: Any, byte_count: Any, name: str) -> int:
    if gib is not None and byte_count is not None:
        raise ValueError(f"{name} 预算不能同时使用 GiB 和字节参数。")
    if byte_count is not None:
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"{name} 字节预算必须是非负整数。")
        return byte_count
    if gib is None or isinstance(gib, bool):
        raise ValueError(f"必须明确提供 {name} 非负预算。")
    try:
        amount = float(gib)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError
        return math.ceil(amount * GIB)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} GiB 预算必须是有限的非负数。") from exc


def _available(resources: dict[str, Any], key: str) -> int:
    value = resources.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} 未知或无效，禁止继续。")
    return value


def gate(
    ram_gib: float | None = None,
    commit_gib: float | None = None,
    *,
    need_ram_bytes: int | None = None,
    need_commit_bytes: int | None = None,
) -> dict[str, Any]:
    """在导入大库/载入图之前调用；预算外 RAM 和 commit 均必须额外保留 3 GiB。"""
    errors: list[str] = []
    requirements: dict[str, int | None] = {"ram": None, "commit": None}
    for name, gib, byte_count in (
        ("ram", ram_gib, need_ram_bytes),
        ("commit", commit_gib, need_commit_bytes),
    ):
        try:
            requirements[name] = _budget_bytes(gib, byte_count, name)
        except ValueError as exc:
            errors.append(str(exc))
    try:
        resources = probe_resources()
    except Exception as exc:
        resources = {"ok": False, "errors": [f"资源采样异常：{exc}"]}
    if resources.get("ok") is not True or resources.get("errors"):
        errors.extend(resources.get("errors") or ["资源采样未成功，禁止放行。"])
    for name in ("ram", "commit"):
        try:
            available = _available(resources, f"{name}_available_bytes")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        required = requirements[name]
        if required is not None and available < required + SYSTEM_RESERVE_BYTES:
            errors.append(
                f"{name} 可用 {available} 字节，小于预算 {required} 字节加固定 3 GiB 系统余量。"
            )
    return {
        "allowed": not errors,
        "errors": errors,
        "resources": resources,
        "need_ram_bytes": requirements["ram"],
        "need_commit_bytes": requirements["commit"],
        "reserve_ram_bytes": SYSTEM_RESERVE_BYTES,
        "reserve_commit_bytes": SYSTEM_RESERVE_BYTES,
    }


class MemoryMonitor:
    """每两秒原子更新资源报告；只请求 runner 在转移边界保存退出，不终止进程。"""

    def __init__(self, path: str | os.PathLike[str], interval: float = 2):
        if isinstance(interval, bool) or not math.isfinite(float(interval)) or interval <= 0:
            raise ValueError("资源采样间隔必须是有限正数。")
        self.path = Path(path)
        self.interval = float(interval)
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._entered = False
        self._children_seen: set[int] = set()
        self._report: dict[str, Any] = {
            "schema_version": 1,
            "root_pid": os.getpid(),
            "active": False,
            "started_at_utc": None,
            "finished_at_utc": None,
            "sample_interval_seconds": self.interval,
            "stop_requested": False,
            "safety_stop_reasons": [],
            "sample_errors": [],
            "write_errors": [],
            "sample_count": 0,
            "min_ram_available_bytes": None,
            "min_commit_available_bytes": None,
            "peak_tree_rss_bytes": None,
            "peak_tree_private_commit_bytes": None,
            "peak_children_snapshot_count": 0,
            "peak_children_snapshot_rss_bytes": None,
            "peak_children_snapshot_private_commit_bytes": None,
            "main_lifetime_peak_working_set_bytes": None,
            "main_lifetime_peak_pagefile_usage_bytes": None,
            "children_seen_pid_count": 0,
            "active_ram_reserve_bytes": SYSTEM_RESERVE_BYTES,
            "active_commit_reserve_bytes": ACTIVE_COMMIT_RESERVE_BYTES,
            "latest": None,
            "measurement_limitations": list(_MEASUREMENT_LIMITATIONS),
        }

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._report["stop_requested"]

    @property
    def report(self) -> dict[str, Any]:
        """返回隔离副本，调用方无法清除已经触发的停止标志。"""
        with self._lock:
            return copy.deepcopy(self._report)

    def _request_stop(self, reason: str) -> None:
        self._report["stop_requested"] = True
        if reason not in self._report["safety_stop_reasons"]:
            self._report["safety_stop_reasons"].append(reason)

    def _extreme(self, field: str, value: int, *, minimum: bool = False) -> None:
        previous = self._report[field]
        if previous is None:
            self._report[field] = value
        else:
            self._report[field] = min(previous, value) if minimum else max(previous, value)

    def _update_measurements(self, resources: dict[str, Any]) -> None:
        if resources.get("ok") is not True or resources.get("errors"):
            errors = resources.get("errors") or ["采样结果未知。"]
            self._report["sample_errors"].extend(errors)
            self._request_stop("资源采样失败或未知，按保守原则请求安全停止。")
        for name, reserve in (
            ("ram", SYSTEM_RESERVE_BYTES),
            ("commit", ACTIVE_COMMIT_RESERVE_BYTES),
        ):
            value = _available(resources, f"{name}_available_bytes")
            self._extreme(f"min_{name}_available_bytes", value, minimum=True)
            if value < reserve:
                self._request_stop(
                    f"系统 {name} 可用量低于安全余量 {reserve} 字节，请在转移边界保存并退出。"
                )
        tree = resources.get("process_tree")
        if not isinstance(tree, dict):
            raise ValueError("当前进程树内存未知。")
        for source, target in (
            ("rss_bytes", "peak_tree_rss_bytes"),
            ("private_commit_bytes", "peak_tree_private_commit_bytes"),
            ("children_snapshot_count", "peak_children_snapshot_count"),
            ("children_snapshot_rss_bytes", "peak_children_snapshot_rss_bytes"),
            ("children_snapshot_private_commit_bytes", "peak_children_snapshot_private_commit_bytes"),
        ):
            self._extreme(target, _available(tree, source))
        main = tree.get("main_process")
        if not isinstance(main, dict):
            raise ValueError("主进程生命周期峰值未知。")
        self._extreme(
            "main_lifetime_peak_working_set_bytes", _available(main, "peak_working_set_bytes")
        )
        self._extreme(
            "main_lifetime_peak_pagefile_usage_bytes", _available(main, "peak_pagefile_usage_bytes")
        )
        pids = tree.get("child_pids")
        if not isinstance(pids, list) or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids
        ):
            raise ValueError("子进程快照 PID 列表未知或无效。")
        self._children_seen.update(pids)
        self._report["children_seen_pid_count"] = len(self._children_seen)

    def _write_report(self) -> None:
        """同目录临时文件加 replace，读者只会看到上次或本次完整 JSON。"""
        temporary: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self._report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
        except Exception as exc:
            message = f"资源报告原子写入失败：{exc}"
            if message not in self._report["write_errors"]:
                self._report["write_errors"].append(message)
            self._request_stop("无法可靠写入资源报告，按保守原则请求安全停止。")
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _sample_once(self) -> None:
        with self._lock:
            self._report["sample_count"] += 1
            try:
                resources = probe_resources()
                self._report["latest"] = resources
                self._update_measurements(resources)
            except Exception as exc:
                self._report["sample_errors"].append(f"资源监视采样失败：{exc}")
                self._request_stop("资源采样失败或未知，按保守原则请求安全停止。")
            self._report["updated_at_utc"] = _now()
            self._write_report()

    def _run(self) -> None:
        while not self._shutdown.wait(self.interval):
            self._sample_once()

    def __enter__(self) -> MemoryMonitor:
        if self._entered:
            raise RuntimeError("同一个资源监视器不能重复进入上下文。")
        self._entered = True
        with self._lock:
            self._report["active"] = True
            self._report["started_at_utc"] = _now()
        # 首次同步采样，使 runner 在执行第一步前便可检查安全停止标志。
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="资源监视器", daemon=True)
        try:
            self._thread.start()
        except Exception as exc:
            with self._lock:
                self._report["sample_errors"].append(f"资源采样线程启动失败：{exc}")
                self._request_stop("资源监视线程不可用，请求安全停止。")
                self._write_report()
            self._thread = None
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._report["active"] = False
            self._report["finished_at_utc"] = _now()
            if exc_value is not None:
                self._report["context_error"] = f"监视上下文异常：{exc_value}"
        # 最后再采一次，保留主进程生命周期峰值以及退出时系统余量。
        self._sample_once()
        return False
