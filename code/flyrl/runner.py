"""四臂共用的像素闭环入口；旧采集缓存和教师模型不进入本模块。"""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GRAPH_SHA256 = "f4d41f011e97e510761d011634891b1c082f61f91463eb586ee9cf8c6371b1d1"
DEFAULTS = {
    "schema_version": 1, "task": "corridor", "arm": "real", "algorithm": "dqn",
    "seed": 41027, "rewire_seed": 17, "projection_seed": 23, "device": "cpu",
    "graph": "doomfly/outputs/doom/malecns_v1/graph.npz",
    "neurons": "doomfly/connectome_data/malecns_v1/normalized/neurons.feather",
    "catalog": "doomfly/connectome_data/malecns_v1/normalized/catalog.feather",
    "annotations": "doomfly/connectome_data/malecns_v1/annotations.feather",
    "output": "experiment/runs", "frontend": "agc_motion", "intervention": "none",
    "scale": 1.5, "bias": 6.0, "threshold": -48.0, "max_turn": 6,
    "gamma": .999, "max_tics": 2000, "hidden": 0,
    "calibration_steps": 128, "max_env_steps": 20000, "wall_hours": 2.,
    "checkpoint_every": 1000, "epsilon_end": .1, "epsilon_decay_steps": 10000,
    "updates_per_step": 1, "learner": {}, "rewards": {},
    "train_seeds": [41027, 41028, 41029], "val_seeds": [51027, 51028],
    "test_seeds": [61027, 61028, 61029],
    "train_layouts": [], "val_layouts": [], "test_layouts": [],
    "curriculum": {"exit": .1, "near_turn": .3, "entry": 1.},
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path=None, overrides=None):
    supplied = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    supplied.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"未知配置项: {sorted(unknown)}")
    c = {**DEFAULTS, **supplied}
    if c["schema_version"] != 1 or c["task"] not in ("corridor", "combat"):
        raise ValueError("不支持的 schema/task")
    if c["arm"] not in ("real", "rewired", "retinal", "pixels") or c["algorithm"] not in ("dqn", "ppo"):
        raise ValueError("未知 arm/algorithm")
    if c["frontend"] not in ("raw", "agc_motion") or c["intervention"] not in ("none", "black", "scramble"):
        raise ValueError("未知视觉配置")
    for key in ("calibration_steps", "max_env_steps", "max_tics", "checkpoint_every", "epsilon_decay_steps", "updates_per_step"):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} 必须是正整数")
    if not math.isfinite(c["wall_hours"]) or c["wall_hours"] <= 0:
        raise ValueError("wall_hours 必须为有限正数")
    if not 0 <= c["epsilon_end"] <= 1 or not 0 <= c["gamma"] <= 1:
        raise ValueError("epsilon/gamma 越界")
    if c["hidden"] not in (0, 32):
        raise ValueError("共同读出仅允许 hidden=0 或32")
    sets = []
    for key in ("train_seeds", "val_seeds", "test_seeds"):
        values = c[key]
        if not values or len(set(values)) != len(values) or any(type(s) is not int or s < 0 for s in values):
            raise ValueError(f"{key} 必须非空、无重复、非负整数")
        sets.append(set(values))
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("train/val/test seeds 必须分离")
    if c["calibration_steps"] < 2 or c["updates_per_step"] > 8:
        raise ValueError("校准至少2帧，每转移最多8次优化")
    from .environment import LayoutConfig, RewardConfig, ActionCodec, corridor_wad_bytes
    ActionCodec(c["task"], c["max_turn"])
    RewardConfig(**c["rewards"])
    if c["task"] == "corridor":
        layouts = []
        for key in ("train_layouts", "val_layouts", "test_layouts"):
            if not c[key]:
                raise ValueError(f"走廊必须显式指定 {key}")
            c[key] = [asdict(LayoutConfig(**item)) for item in c[key]]
            layouts.append({hashlib.sha256(corridor_wad_bytes(LayoutConfig(**item))).hexdigest() for item in c[key]})
        if any(layouts[i] & layouts[j] for i in range(3) for j in range(i)):
            raise ValueError("layout 配置不可跨 split 完全重复")
        schedule = c["curriculum"]
        if (set(schedule) != {"exit", "near_turn", "entry"} or
                not 0 < schedule["exit"] < schedule["near_turn"] < schedule["entry"] == 1):
            raise ValueError("curriculum 必须是 exit < near_turn < entry=1 的累计比例")
        if {x["layout_id"] for x in c["train_layouts"]} != set(schedule):
            raise ValueError("train_layouts 须覆盖全部课程阶段")
        if c["calibration_steps"] < 2 * len(c["train_layouts"]):
            raise ValueError("校准至少为每个训练布局分配一帧初始画面和一次随机转移")
    elif any(c[key] for key in ("train_layouts", "val_layouts", "test_layouts")):
        raise ValueError("combat 尚无地图变体，不得声称布局泛化")
    from .learning import DQNConfig, PPOConfig
    cls = DQNConfig if c["algorithm"] == "dqn" else PPOConfig
    reserved = {"obs_dim", "n_actions", "hidden", "device", "seed", "gamma"}
    if reserved & set(c["learner"]):
        raise ValueError("learner 不得覆盖共同容量、种子、设备或 gamma")
    cls(obs_dim=246, n_actions=6 if c["task"] == "corridor" else 12,
        hidden=c["hidden"], device=c["device"], seed=c["seed"], gamma=c["gamma"], **c["learner"])
    return c


def provenance(c):
    versions = {"python": sys.version, "platform": sys.platform}
    for name in ("numpy", "numba", "torch", "vizdoom", "pyarrow", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    paths = {key: resolve(c[key]) for key in ("graph", "neurons", "catalog", "annotations")}
    import vizdoom
    paths["iwad"] = Path(vizdoom.__file__).parent / "freedoom2.wad"
    if c["task"] == "combat":
        paths["combat_wad"] = ROOT / "doomfly/doom/scenarios/combat_survival.wad"
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob("*.py")}
    sources.update({str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "doomfly/doom").glob("*.py")})
    return {"versions": versions, "inputs": {key: {"path": str(p), "sha256": sha256(p)} for key, p in paths.items()},
            "sources": sources, "argv": sys.argv, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def new_run(c, kind):
    path = resolve(c["output"]) / (time.strftime("%Y%m%d-%H%M%S") + f"-{kind}-{c['arm']}-" + uuid.uuid4().hex[:10])
    path.mkdir(parents=True, exist_ok=False)
    json_write(path / "config.json", c)
    return path


def anatomical_sides(path, graph_ids):
    """精确ID对齐显式侧别；保留中线，冲突为unknown，不从类型名称猜测。"""
    import pandas as pd
    table = pd.read_feather(path, columns=["bodyId", "somaSide", "rootSide"])
    if table["bodyId"].duplicated().any() or table["bodyId"].dtype.kind not in "iu":
        raise ValueError("原始侧别 bodyId 必须是唯一整数")
    table = table.set_index("bodyId").loc[np.asarray(graph_ids)]
    mapping, conflicts = {}, []
    for ident, soma, root in table.itertuples():
        s = "" if pd.isna(soma) else str(soma).strip()
        t = "" if pd.isna(root) else str(root).strip()
        if s in ("L", "R") and t in ("L", "R") and s != t:
            side = "unknown"
            conflicts.append(int(ident))
        elif s in ("L", "R"):
            side = s
        elif not s and t in ("L", "R"):
            side = t
        else:
            side = "unknown"
        mapping[int(ident)] = side
    report = {"source": str(path), "policy": "somaSide L/R；仅缺失时rootSide L/R；冲突/中线unknown",
              "counts": {side: sum(x == side for x in mapping.values()) for side in ("L", "R", "unknown")},
              "conflicts": conflicts}
    return mapping, report


class Pipeline:
    """同维同 trace 的四臂；baseline 不经过任何全脑传播或可训练 CNN。"""
    dim = 246

    def __init__(self, c):
        from .encoding import Vision, build_groups, MultiscaleRates
        self.c, self.brain = c, None
        with np.load(resolve(c["graph"]), allow_pickle=False) as data:
            uv, ids = data["uv"], data["ids"]
        sides, self.side_report = anatomical_sides(resolve(c["annotations"]), ids)
        self.groups = build_groups(ids, resolve(c["neurons"]), resolve(c["catalog"]), side_by_id=sides)
        self.groups.provenance.update(side_source=self.side_report)
        self.group_signature = signature({"names": self.groups.group_names,
            "indices": [idx.tolist() for idx in self.groups.group_indices]})
        self.vision = Vision(uv, c["frontend"], c["intervention"], c["projection_seed"])
        if c["arm"] in ("real", "rewired"):
            from .brain import FrozenBrain
            self.brain = FrozenBrain(resolve(c["graph"]), scale=c["scale"], bias=c["bias"],
                                     threshold=c["threshold"], rewired=c["arm"] == "rewired", rewire_seed=c["rewire_seed"])
            self.rates = MultiscaleRates(self.groups)
        else:
            # 3×41 固定空间池；retinal 按同一受体 UV，pixels 直接对线性亮度像素池化。
            self.bins = np.minimum((uv[:, 1] * 3).astype(int), 2) * 41 + np.minimum((uv[:, 0] * 41).astype(int), 40)
            self.bin_count = np.bincount(self.bins, minlength=123)
        self.mu = np.zeros(self.dim, np.float32)
        self.sd = np.ones(self.dim, np.float32)
        self.calibrated = False
        self.reset()

    def reset(self):
        self.vision.reset()
        if self.brain is not None:
            self.brain.reset()
            self.rates.reset()
        self.trace = np.zeros((2, 123), np.float64)
        self.pixel_previous = None

    def raw(self, rgb):
        duration = 1000 / 35
        if self.brain is not None:
            result = self.brain.step(self.vision.step(rgb), duration)
            value = self.rates.step(result.counts, result.actual_ms)
            self.last_diagnostic = {"actual_ms": result.actual_ms, "brain_wall_s": result.wall_s,
                                    "spikes": int(result.counts.sum()), "active": int(self.brain.nactive[0])}
        else:
            if self.c["arm"] == "retinal":
                lum = self.vision.step(rgb)
                base = np.bincount(self.bins, weights=lum, minlength=123) / np.maximum(self.bin_count, 1)
            else:
                # 固定像素网格，完整覆盖每个池；与受体采样路径不同，不使用学习投影。
                p = rgb.astype(np.float32) / 255
                p = np.where(p <= .04045, p / 12.92, ((p + .055) / 1.055) ** 2.4)
                lum = p @ np.asarray([.2126, .7152, .0722], np.float32)
                if self.c["intervention"] == "black":
                    lum.fill(0)
                elif self.c["intervention"] == "scramble":
                    rng = np.random.default_rng(self.c["projection_seed"])
                    lum = lum.ravel()[rng.permutation(lum.size)].reshape(lum.shape)
                out = lum.copy()
                if self.c["frontend"] == "agc_motion" and self.c["intervention"] != "black":
                    if self.pixel_previous is not None:
                        out = np.clip(lum + 1.2 * (lum - self.pixel_previous), 0, 1)
                    sd = out.std()
                    if sd >= 1e-4:
                        out = .5 + .5 * np.tanh((out - out.mean()) / (sd + 1e-4) / 1.5)
                self.pixel_previous = lum.copy()
                base = np.array([block.mean() for row in np.array_split(out, 3, axis=0)
                                 for block in np.array_split(row, 41, axis=1)])
            alpha = -np.expm1(-duration / np.array([50., 200.]))
            # 代理亮度单位不是生理 Hz；共同 log1p/trace 后用各臂训练随机轨迹统计归一化。
            self.trace += alpha[:, None] * (1000 * base[None, :] - self.trace)
            value = np.log1p(self.trace).ravel().astype(np.float32)
            self.last_diagnostic = {"actual_ms": duration, "brain_wall_s": 0., "spikes": None}
        if value.shape != (self.dim,) or not np.isfinite(value).all():
            raise ValueError("非有限或错误维度的策略特征")
        return value

    def observe(self, rgb):
        return ((self.raw(rgb) - self.mu) / self.sd).clip(-10, 10).astype(np.float32)

    def fit(self, features):
        data = np.asarray(features, np.float64)
        if data.ndim != 2 or len(data) < 2 or data.shape[1] != self.dim or not np.isfinite(data).all():
            raise ValueError("校准须至少两帧有限训练特征")
        self.mu = data.mean(0).astype(np.float32)
        self.sd = np.maximum(data.std(0), .01).astype(np.float32)
        self.calibrated = True

    def save(self, path):
        import torch
        torch.save({"schema": 1, "feature_version": "groups123-anatomical-sides-trace50-200-v2", "config": self.c,
                    "group_signature": self.group_signature,
                    "mu": torch.from_numpy(self.mu), "sd": torch.from_numpy(self.sd), "calibrated": self.calibrated}, path)

    def load(self, path):
        import torch
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["schema"] != 1 or data["feature_version"] != "groups123-anatomical-sides-trace50-200-v2" or data["config"] != self.c:
            raise ValueError("pipeline schema/特征语义/配置不匹配")
        if data["group_signature"] != self.group_signature:
            raise ValueError("解剖群体索引与checkpoint不一致")
        self.mu, self.sd = data["mu"].numpy().copy(), data["sd"].numpy().copy()
        if self.mu.shape != (self.dim,) or self.sd.shape != (self.dim,) or not np.isfinite(self.mu).all() or not np.isfinite(self.sd).all() or np.any(self.sd <= 0):
            raise ValueError("checkpoint 归一化统计无效")
        self.calibrated = bool(data["calibrated"])


def make_env(c, run, split, episode, rng, step=0, calibration=False):
    from .environment import GameEnv
    layout = None
    if c["task"] == "corridor":
        choices = c[f"{split}_layouts"]
        if split == "train" and not calibration:
            fraction = min(step / c["max_env_steps"], 1)
            stage = next(key for key in ("exit", "near_turn", "entry") if fraction <= c["curriculum"][key])
            choices = [x for x in choices if x["layout_id"] == stage]
        layout = choices[int(rng.integers(len(choices)))] if split == "train" and not calibration else choices[episode % len(choices)]
    seeds = c[f"{split}_seeds"]
    if split != "train":
        ncases = len(c[f"{split}_layouts"]) if c["task"] == "corridor" else len(seeds)
        if episode >= ncases:
            raise ValueError("评测超过唯一layout（combat为seed）数；拒绝伪重复")
        seed = seeds[episode % len(seeds)]
    else:
        seed = seeds[episode % len(seeds)]
    return GameEnv(c["task"], seed, c["max_tics"], layout,
                   max_turn=c["max_turn"], gamma=c["gamma"], rewards=c["rewards"],
                   run_dir=run / "assets" / f"{split}-{episode:06d}", upstream_root=ROOT / "doomfly")


def record(path, value):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def calibrate(c, pipeline, run, deadline):
    rng = np.random.default_rng(c["seed"] + 100000)
    features, episode, env = [], 0, None
    episodes, interactions = [], 0
    try:
        while len(features) < c["calibration_steps"]:
            if time.perf_counter() >= deadline:
                raise TimeoutError("校准耗尽 wall-time；尚未启动正式训练")
            env = make_env(c, run, "train", episode, rng, calibration=True)
            rgb, initial_info = env.reset()
            episodes.append({"seed": initial_info.get("seed"), "layout": initial_info.get("layout"),
                             "assets": initial_info.get("assets"), "actions": []})
            pipeline.reset()
            features.append(pipeline.raw(rgb))
            nlayouts = max(len(c["train_layouts"]), 1)
            quota = c["calibration_steps"] // nlayouts + int(episode % nlayouts < c["calibration_steps"] % nlayouts)
            for _ in range(min(quota - 1, c["calibration_steps"] - len(features))):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("校准耗尽 wall-time")
                action = int(rng.integers(env.n_actions))
                episodes[-1]["actions"].append(action)
                interactions += 1
                rgb, _, terminated, truncated, _ = env.step(action)
                if rgb is not None:
                    features.append(pipeline.raw(rgb))
                if terminated or truncated:
                    break
            env.close()
            episode += 1
        pipeline.fit(features)
        json_write(run / "calibration.json", {"frames": len(features), "interactions": interactions,
                                             "episodes": episodes, "split": "train", "policy": "uniform_random",
                                             "action_labels_used": False, "constant_dimensions": int(np.count_nonzero(np.std(features, axis=0) < 1e-6))})
        return len(features)
    finally:
        if env is not None:
            env.close()
        pipeline.reset()


def learner(c):
    from .learning import DQNAgent, PPOAgent
    cls = DQNAgent if c["algorithm"] == "dqn" else PPOAgent
    return cls(obs_dim=246, n_actions=6 if c["task"] == "corridor" else 12,
               hidden=c["hidden"], device=c["device"], seed=c["seed"], gamma=c["gamma"], **c["learner"])


def checkpoint(run, c, pipeline, agent, steps, prov):
    folder = run / (f"checkpoint-{steps:08d}-" + uuid.uuid4().hex[:8])
    folder.mkdir(exist_ok=False)
    pipeline.save(folder / "pipeline.pt")
    agent.save(folder / "learner.pt")
    json_write(folder / "manifest.json", {"schema": 1, "config": c, "provenance": prov, "steps": steps,
                                         "files": {name: sha256(folder / name) for name in ("pipeline.pt", "learner.pt")},
                                         "resume": False, "note": "无游戏/脑动态状态快照；只供冻结评测，不声称精确续训"})
    json_write(run / "latest-checkpoint.json", {"checkpoint": str(folder)})
    return folder


def train(c, *, purpose="train"):
    start = time.perf_counter()
    deadline = start + c["wall_hours"] * 3600
    if purpose not in ("train", "integration-test"):
        raise ValueError("未知运行目的")
    run = new_run(c, purpose)
    prov = provenance(c)
    json_write(run / "provenance.json", prov)
    pipeline, agent = Pipeline(c), learner(c)
    pipeline.groups.save(run / "groups.npz")
    if pipeline.brain is not None and pipeline.brain.rewire_report is not None:
        json_write(run / "rewire.json", pipeline.brain.rewire_report)
    steps, episode, calibration_frames, env = 0, 0, 0, None
    rng = np.random.default_rng(c["seed"])
    rollout = {k: [] for k in ("obs", "actions", "rewards", "old_logp", "values", "next_values", "terminated", "truncated")}
    latest = None
    try:
        calibration_frames = calibrate(c, pipeline, run, deadline)
        while steps < c["max_env_steps"] and time.perf_counter() < deadline:
            env = make_env(c, run, "train", episode, rng, steps)
            rgb, info = env.reset()
            pipeline.reset()
            obs = pipeline.observe(rgb)
            ep_return, ep_steps, max_progress = 0., 0, 0.
            action_counts = np.zeros(env.n_actions, np.int64)
            ended = False
            while steps < c["max_env_steps"] and time.perf_counter() < deadline:
                if c["algorithm"] == "dqn":
                    epsilon = 1 - (1 - c["epsilon_end"]) * min(steps / c["epsilon_decay_steps"], 1)
                    action = agent.act(obs, epsilon)
                else:
                    action, logp, value = agent.act(obs)
                action_counts[action] += 1
                rgb, reward, terminated, truncated, info = env.step(action)
                steps += 1
                ep_steps += 1
                ep_return += reward
                max_progress = max(max_progress, info.get("progress") or 0.)
                nxt = np.zeros(246, np.float32) if terminated else pipeline.observe(rgb)
                metrics = None
                if c["algorithm"] == "dqn":
                    agent.remember(obs, action, reward, nxt, terminated=terminated, truncated=truncated)
                    for _ in range(c["updates_per_step"]):
                        if time.perf_counter() >= deadline:
                            break
                        metrics = agent.update()
                else:
                    row = (obs, action, reward, logp, value, 0. if terminated else agent.value(nxt), terminated, truncated)
                    for key, item in zip(rollout, row):
                        rollout[key].append(item)
                    if len(rollout["obs"]) >= agent.config.rollout_steps and time.perf_counter() < deadline:
                        metrics = agent.update(rollout)
                        rollout = {k: [] for k in rollout}
                if metrics is not None and (steps % 25 == 0 or c["algorithm"] == "ppo"):
                    record(run / "updates.jsonl", {"step": steps, **metrics})
                obs = nxt
                if steps % c["checkpoint_every"] == 0:
                    latest = checkpoint(run, c, pipeline, agent, steps, prov)
                if terminated or truncated:
                    ended = True
                    break
            record(run / "episodes.jsonl", {"episode": episode, "step": steps, "return": ep_return,
                    "episode_steps": ep_steps, "complete": ended, "success": info["success"],
                    "action_counts": action_counts.tolist(), "policy_entropy_proxy": action_entropy(action_counts),
                    "max_progress": max_progress, "info": {k: v for k, v in info.items() if k != "final_observation"}})
            env.close()
            episode += 1
        if c["algorithm"] == "ppo" and rollout["obs"] and time.perf_counter() < deadline:
            record(run / "updates.jsonl", {"step": steps, **agent.update(rollout)})
        collection_wall_s = time.perf_counter() - start
        # 最终更新后总是产生新快照，不覆盖周期checkpoint；持久化耗时另记。
        latest = checkpoint(run, c, pipeline, agent, steps, prov)
        result = {"status": "budget_complete", "purpose": purpose, "steps": steps, "calibration_frames": calibration_frames,
                  "wall_s": time.perf_counter() - start, "collection_wall_s": collection_wall_s,
                  "checkpoint": str(latest), "run": str(run),
                  "budget_granularity": "一个env+编码/优化调用不可中断；截止后不再启动优化，最终持久化另计",
                  "stop": "env_step_limit" if steps >= c["max_env_steps"] else "wall_time_limit"}
        json_write(run / "result.json", result)
        return result
    except BaseException as exc:
        json_write(run / "failure.json", {"type": type(exc).__name__, "message": str(exc), "steps": steps,
                                         "wall_s": time.perf_counter() - start})
        raise
    finally:
        if env is not None:
            env.close()


def smoke(c, steps=4):
    if not 1 <= steps <= 128:
        raise ValueError("smoke steps 必须1..128；不是正式训练")
    start = time.perf_counter()
    run = new_run(c, "smoke")
    prov = provenance(c)
    json_write(run / "provenance.json", prov)
    pipeline = Pipeline(c)
    pipeline.groups.save(run / "groups.npz")
    rng = np.random.default_rng(c["seed"])
    frames, env, info = [], None, None
    try:
        env = make_env(c, run, "train", 0, rng)
        rgb, info = env.reset()
        pipeline.reset()
        obs = pipeline.raw(rgb)
        agent = learner(c)
        frames.append({"tic": 0, "feature_norm": float(np.linalg.norm(obs)), **pipeline.last_diagnostic})
        for i in range(steps):
            action = agent.act(obs) if c["algorithm"] == "dqn" else agent.act(obs, deterministic=True)[0]
            rgb, reward, terminated, truncated, info = env.step(action)
            if rgb is not None:
                obs = pipeline.raw(rgb)
                frames.append({"tic": i + 1, "feature_norm": float(np.linalg.norm(obs)), **pipeline.last_diagnostic})
            if terminated or truncated:
                break
        # 仅保存/重载未训练头，验证 checkpoint 格式，不执行优化。
        folder = checkpoint(run, c, pipeline, agent, 0, prov)
        type(agent).from_checkpoint(folder / "learner.pt", device=c["device"])
        pipeline.load(folder / "pipeline.pt")
        graph_unchanged = sha256(resolve(c["graph"])) == prov["inputs"]["graph"]["sha256"]
        result = {"run": str(run), "trained": False, "frames": frames, "wall_s": time.perf_counter() - start,
                  "graph_unchanged": graph_unchanged, "groups": len(pipeline.groups.group_names),
                  "empty_groups": int(np.count_nonzero(pipeline.groups.group_counts == 0)),
                  "side_report": pipeline.side_report, "rewire_report": pipeline.brain.rewire_report if pipeline.brain is not None else None,
                  "feature_dim": 246, "checkpoint_reload": True,
                  "assets": info["assets"], "layout": info["layout"]}
        json_write(run / "result.json", result)
        if not graph_unchanged:
            raise RuntimeError("脑图哈希变化")
        return result
    finally:
        if env is not None:
            env.close()


def evaluate(folder, split="val", episodes=None, *, intervention="none", policy="learned"):
    if split not in ("val", "test") or (episodes is not None and episodes < 1):
        raise ValueError("仅允许 val/test，episodes>0")
    if intervention not in ("none", "black", "scramble") or policy not in ("learned", "random", "untrained"):
        raise ValueError("未知评估输入干预/策略")
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != 1:
        raise ValueError("不支持的checkpoint schema")
    for name, digest in manifest["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"checkpoint 内容哈希不符: {name}")
    c = load_config(overrides=manifest["config"])
    ncases = len(c[f"{split}_layouts"]) if c["task"] == "corridor" else len(c[f"{split}_seeds"])
    episodes = ncases if episodes is None else episodes
    if episodes > ncases:
        raise ValueError("评测超过唯一 seed×layout 组合数，不把重复确定性轨迹计为独立样本")
    current = provenance(c)
    if (current["inputs"] != manifest["provenance"]["inputs"] or
            current["sources"] != manifest["provenance"]["sources"] or
            current["versions"] != manifest["provenance"]["versions"]):
        raise ValueError("源码/输入与训练记录不一致；拒绝静默改变实验")
    run = new_run(c, f"eval-{split}")
    pipeline = Pipeline(c)
    pipeline.load(folder / "pipeline.pt")
    if not pipeline.calibrated:
        raise ValueError("smoke checkpoint 未校准，不能当正式模型评估")
    pipeline.c = {**c, "intervention": intervention}
    pipeline.vision.intervention = intervention
    agent = learner(c)
    if policy == "learned":
        agent.load(folder / "learner.pt")
    rows, env = [], None
    rng = np.random.default_rng(c["seed"] + 200000)
    start = time.perf_counter()
    deadline = start + c["wall_hours"] * 3600
    try:
        for episode in range(episodes):
            if time.perf_counter() >= deadline:
                raise TimeoutError("评估超出 wall-time；不得以部分局冒充完整评估")
            env = make_env(c, run, split, episode, rng)
            rgb, info = env.reset()
            pipeline.reset()
            obs, total, maximum, action_counts = pipeline.observe(rgb), 0., 0., np.zeros(env.n_actions, int)
            while True:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("评估超出 wall-time")
                if policy == "random":
                    action = int(rng.integers(env.n_actions))
                else:
                    action = agent.act(obs, epsilon=0) if c["algorithm"] == "dqn" else agent.act(obs, deterministic=True)[0]
                action_counts[action] += 1
                rgb, reward, terminated, truncated, info = env.step(action)
                total += reward
                maximum = max(maximum, info.get("progress") or 0.)
                if terminated or truncated:
                    break
                obs = pipeline.observe(rgb)
            row = {"episode": episode, "return": total, "max_progress": maximum, "success": info["success"],
                   "tics": info["tics"], "wall_hits": info["wall_hits"], "action_counts": action_counts.tolist(),
                   "telemetry": info["telemetry"], "layout": info["layout"], "assets": info["assets"], "seed": info["seed"]}
            rows.append(row)
            record(run / "episodes.jsonl", row)
            env.close()
        result = {"run": str(run), "checkpoint": str(folder.resolve()), "checkpoint_manifest_sha256": sha256(folder / "manifest.json"),
                  "config": c, "split": split, "trained": False, "conditions": {"intervention": intervention, "policy": policy},
                  "training_steps": manifest["steps"],
                  "calibration_frames": c["calibration_steps"], "episodes": rows,
                  "summary": summarize(rows), "wall_s": time.perf_counter() - start,
                  "sources": current["sources"], "inputs": current["inputs"], "versions": current["versions"]}
        json_write(run / "result.json", result)
        return result
    finally:
        if env is not None:
            env.close()


def action_entropy(counts):
    counts = np.asarray(counts, np.float64)
    probabilities = counts[counts > 0] / max(counts.sum(), 1)
    return float(-(probabilities * np.log(probabilities)).sum())


def summarize(rows):
    success = sum(x["success"] for x in rows)
    n = len(rows)
    z = 1.95996398454
    p = success / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    returns = np.array([x["return"] for x in rows])
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(returns, n, replace=True).mean() for _ in range(2000)])
    return {"n": n, "success_rate": p, "success_wilson95": [center-half, center+half],
            "return_mean": float(returns.mean()), "return_bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "mean_tics": float(np.mean([x["tics"] for x in rows])),
            "note": "episode 条件置信区间不替代多训练 seed / 多重连实例的不确定性"}


def compare(paths):
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if len(rows) < 2:
        raise ValueError("至少提供两个评估result.json")
    omit = {"arm", "rewire_seed", "seed", "output", "device", "scale", "bias", "threshold"}
    # 动力学参数也必须共同；即便 baseline 不使用这些参数也不能混淆实验版本。
    omit -= {"scale", "bias", "threshold"}
    def comparable(row):
        return {"config": {k: v for k, v in row["config"].items() if k not in omit},
                "split": row["split"], "conditions": row.get("conditions", {"intervention":"none", "policy":"learned"}),
                "steps": row["training_steps"], "calibration": row["calibration_frames"],
                "sources": row["sources"], "inputs": row["inputs"], "versions": row["versions"],
                "episodes": [(x["seed"], x["layout"], x["assets"]) for x in row["episodes"]]}
    if any(comparable(x) != comparable(rows[0]) for x in rows[1:]):
        raise ValueError("任务、交互预算、布局、特征、算法或源码不可比；拒绝汇总")
    return {"comparable": True, "runs": [{"arm": r["config"]["arm"], "seed": r["config"]["seed"],
             "rewire_seed": r["config"]["rewire_seed"], "checkpoint": r["checkpoint"], "summary": r["summary"]} for r in rows],
            "claim": "单次成绩只作pilot，不能推出连接组优势"}


def doctor(c):
    import torch
    prov = provenance(c)
    digest = prov["inputs"]["graph"]["sha256"]
    if digest != GRAPH_SHA256:
        raise ValueError("权威脑图 SHA-256 不匹配；绝不自动覆盖")
    return {"versions": prov["versions"], "graph_sha256": digest, "graph_baseline_matches": True,
            "cuda_available": torch.cuda.is_available(), "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "simulation_backend": "CPU Numba；CUDA仅用于外部小读出", "formal_training_started": False}
