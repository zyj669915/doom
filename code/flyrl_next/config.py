"""Strict, dependency-light configuration and train-only success-gated curriculum."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from flyrl.environment import LayoutConfig, RewardConfig, corridor_wad_bytes

ROOT = Path(__file__).resolve().parents[3]
ROUTE_ALGORITHMS = {"A1": "dqn", "A2": "dqn", "A3": "dqn", "A4": "ppo",
                    "A5": "ppo", "A6": "dqn", "A7": "ppo"}
STAGES = ("exit", "near_turn", "entry")
GATE_DEFAULTS = {"window": 20, "min_episodes": 10, "success_rate": .6,
                 "min_side_success": 2, "easy_mix": .2}


def _layouts(split, count):
    offset = {"train": 1000, "val": 2000, "test": 3000}[split]
    return [asdict(LayoutConfig(layout_id=STAGES[(i // 2) % 3] if split == "train" else "entry",
                               layout_seed=offset + i, mirror="left" if i % 2 == 0 else "right",
                               lightlevel={"train": 224, "val": 200, "test": 176}[split]))
            for i in range(count)]


DEFAULTS = {
    "schema_version": 1, "route": "A3", "algorithm": "dqn", "seed": 41027,
    "device": "cpu", "task": "corridor", "action_repeat": 4, "action_set": "full",
    "gamma_tic": .999, "max_tics": 2000, "max_game_tics": 10000000,
    "wall_hours": 2., "checkpoint_every": 2000, "calibration_steps": 128,
    "output": "experiment/runs_next",
    "graph": "doomfly/outputs/doom/malecns_v1/graph.npz",
    "neurons": "doomfly/connectome_data/malecns_v1/normalized/neurons.feather",
    "catalog": "doomfly/connectome_data/malecns_v1/normalized/catalog.feather",
    "annotations": "doomfly/connectome_data/malecns_v1/annotations.feather",
    "frontend": "agc_motion", "scale": 1.5, "bias": 6., "threshold": -48.,
    "hidden": 0, "max_turn": 6., "rewire_seed": 17, "projection_seed": 23,
    "learner": {}, "rollout_steps": 256, "epsilon_end": .1, "epsilon_decay_steps": 10000,
    "rewards": asdict(RewardConfig()), "objective": "potential",
    "curriculum": "gated", "curriculum_gate": GATE_DEFAULTS,
    "curriculum_schedule": {"exit": .1, "near_turn": .3, "entry": 1.},
    "train_layouts": _layouts("train", 12), "val_layouts": _layouts("val", 12),
    "test_layouts": _layouts("test", 24),
    "train_seeds": [41027, 41028, 41029], "val_seeds": [51027, 51028, 51029],
    "test_seeds": [61027, 61028, 61029],
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _unknown(values, allowed, name):
    if not isinstance(values, dict):
        raise ValueError(f"{name} 必须是对象")
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"{name} 未知字段: {sorted(unknown)}")


def _number(value, name, low=None, high=None, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")
    if integer and type(value) is not int:
        raise ValueError(f"{name} 必须是整数")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} 越界 [{low}, {high}]")


def _learner(values, algorithm):
    from dataclasses import fields
    from .learning import DQNConfig, PPOConfig
    cls = DQNConfig if algorithm == "dqn" else PPOConfig
    _unknown(values, {f.name for f in fields(cls)} - {"device", "seed"}, "learner")
    integers = {"batch_size", "replay_capacity", "learning_starts", "target_every", "epochs", "minibatch_size"}
    for key, value in values.items():
        if key == "normalize_advantage":
            if type(value) is not bool:
                raise ValueError("normalize_advantage 必须是布尔值")
        elif key == "critic_hidden":
            if type(value) is not int or value not in (0, 32):
                raise ValueError("critic_hidden 仅允许0或32")
        elif key == "target_kl" and value is None:
            continue
        elif key in integers:
            _number(value, key, 0 if key == "learning_starts" else 1, integer=True)
        elif key in {"gae_lambda", "clip_ratio"}:
            _number(value, key, 0, 1)
        elif key in {"lr", "max_grad_norm", "target_kl"}:
            _number(value, key, 1e-15)
        else:
            _number(value, key, 0)
    if algorithm == "dqn":
        capacity = values.get("replay_capacity", 20000)
        if capacity > 20000 or max(values.get("batch_size", 64), values.get("learning_starts", 256)) > capacity:
            raise ValueError("replay容量上限20000，且须容纳batch与warmup")
    cls(**values)


def load_config(path=None, overrides=None):
    """Return a detached complete dict; paths are relative to ROOT, not config location.

    Nested learner/rewards/gate overrides merge by key. None overrides are ignored
    for CLI compatibility. Unknown fields, including nested ones, are errors.
    """
    supplied = {} if path is None else json.loads(resolve(path).read_text(encoding="utf-8-sig"))
    _unknown(supplied, DEFAULTS, "config")
    overrides = {} if overrides is None else overrides
    _unknown(overrides, DEFAULTS, "overrides")
    c = deepcopy(DEFAULTS)
    for source in (supplied, overrides):
        for key, value in source.items():
            if value is None:
                continue
            if key in ("learner", "rewards", "curriculum_gate", "curriculum_schedule"):
                if not isinstance(value, dict):
                    raise ValueError(f"{key} 必须是对象")
                c[key].update(deepcopy(value))
            else:
                c[key] = deepcopy(value)
    if c["route"] not in ROUTE_ALGORITHMS:
        raise ValueError("route 必须为A1..A7")
    explicit_algorithm = overrides.get("algorithm") or supplied.get("algorithm")
    if explicit_algorithm is None:
        c["algorithm"] = ROUTE_ALGORITHMS[c["route"]]
    if c["algorithm"] != ROUTE_ALGORITHMS[c["route"]]:
        raise ValueError("route 与 algorithm 不匹配")
    if type(c["schema_version"]) is not int or c["schema_version"] != 1 or c["task"] != "corridor":
        raise ValueError("仅支持schema_version=1、task=corridor")
    for key, options in {"device": ("cpu", "cuda"), "frontend": ("agc_motion", "raw", "adapt", "contrast"),
                         "action_set": ("full", "forward"), "objective": ("potential", "progress_delta"),
                         "curriculum": ("gated", "fixed")}.items():
        if c[key] not in options:
            raise ValueError(f"未知{key}: {c[key]}")
    for key in ("seed", "rewire_seed", "projection_seed"):
        _number(c[key], key, 0, integer=True)
    for key in ("action_repeat", "max_tics", "max_game_tics", "checkpoint_every", "calibration_steps", "rollout_steps", "epsilon_decay_steps"):
        _number(c[key], key, 1, integer=True)
    _number(c["calibration_steps"], "calibration_steps", 2, integer=True)
    _number(c["wall_hours"], "wall_hours", 1e-15)
    _number(c["gamma_tic"], "gamma_tic", 0, 1)
    _number(c["epsilon_end"], "epsilon_end", 0, 1)
    _number(c["scale"], "scale", 0)
    _number(c["bias"], "bias")
    _number(c["threshold"], "threshold", -52, -1e-15)
    _number(c["max_turn"], "max_turn", 1e-15, 15)
    if type(c["hidden"]) is not int or c["hidden"] not in (0, 32):
        raise ValueError("hidden 仅允许0或32")
    for key in ("output", "graph", "neurons", "catalog", "annotations"):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError(f"{key} 必须为非空路径字符串")
    seed_sets = []
    for key in ("train_seeds", "val_seeds", "test_seeds"):
        values = c[key]
        if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"{key} 必须为非空非负整数列表")
        if len(values) != len(set(values)):
            raise ValueError(f"{key} 不得重复")
        seed_sets.append(set(values))
    if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("train/val/test seeds 必须互斥")
    hashes = set()
    for key in ("train_layouts", "val_layouts", "test_layouts"):
        if not isinstance(c[key], list) or not c[key]:
            raise ValueError(f"{key} 必须非空")
        normalized = []
        for item in c[key]:
            _unknown(item, LayoutConfig.__dataclass_fields__, key)
            for name in ("layout_seed", "lightlevel"):
                if name in item:
                    _number(item[name], name, 0, integer=True)
            for name in ("spawn_jitter", "angle_jitter"):
                if name in item:
                    _number(item[name], name, 0)
            layout = LayoutConfig(**item)
            digest = hashlib.sha256(corridor_wad_bytes(layout)).hexdigest()
            if digest in hashes:
                raise ValueError("真实WAD内容hash同split或跨split重复")
            hashes.add(digest)
            normalized.append(asdict(layout))
        c[key] = normalized
    if {(x["layout_id"], x["mirror"]) for x in c["train_layouts"]} != {
            (stage, side) for stage in STAGES for side in ("left", "right")}:
        raise ValueError("train_layouts 必须覆盖全部阶段的左右布局")
    if c["calibration_steps"] < 2 * len(c["train_layouts"]):
        raise ValueError("校准至少覆盖每训练布局的初始帧与一次随机转移")
    _unknown(c["rewards"], RewardConfig.__dataclass_fields__, "rewards")
    for key, value in c["rewards"].items():
        _number(value, "rewards." + key)
    RewardConfig(**c["rewards"])
    _learner(c["learner"], c["algorithm"])
    gate = c["curriculum_gate"]
    _unknown(gate, GATE_DEFAULTS, "curriculum_gate")
    for key in ("window", "min_episodes", "min_side_success"):
        _number(gate[key], key, 1, integer=True)
    for key in ("success_rate", "easy_mix"):
        _number(gate[key], key, 0, 1)
    if gate["min_episodes"] > gate["window"] or 2 * gate["min_side_success"] > gate["window"]:
        raise ValueError("课程门限超过窗口大小")
    schedule = c["curriculum_schedule"]
    _unknown(schedule, STAGES, "curriculum_schedule")
    for key in STAGES:
        _number(schedule[key], "curriculum_schedule." + key, 0, 1)
    if not 0 < schedule["exit"] < schedule["near_turn"] < schedule["entry"] == 1:
        raise ValueError("固定课程必须exit < near_turn < entry=1")
    return c


def _identity(layout):
    if isinstance(layout, LayoutConfig):
        layout = asdict(layout)
    return json.dumps(asdict(LayoutConfig(**layout)), sort_keys=True)


class Curriculum:
    """Gates use training outcomes only, with no step-count fallback.

    At each stage the latest <=20 training episodes must include >=10 outcomes,
    >=60% successes and >=2 successes on each side. After unlocking a stage,
    20% of selections rehearse easier starts. Fixed mode follows the original
    10%/30%/100% cumulative schedule; advance() receives actual game tics.
    """
    def __init__(self, config):
        self.config = load_config(overrides=config)
        self.gate = self.config["curriculum_gate"]
        self.stage = 0
        self.game_tics = 0
        self.layouts = deepcopy(self.config["train_layouts"])
        self.by_key = {_identity(x): x for x in self.layouts}
        self.history = {s: deque(maxlen=self.gate["window"]) for s in STAGES}

    @property
    def stage_name(self):
        return STAGES[self.stage]

    def advance(self, game_tics):
        """Add actual training game tics (not decisions); no success fallback."""
        _number(game_tics, "game_tics", 0, integer=True)
        self.game_tics += game_tics
        if self.config["curriculum"] == "fixed":
            fraction = self.game_tics / self.config["max_game_tics"]
            schedule = self.config["curriculum_schedule"]
            self.stage = 0 if fraction < schedule["exit"] else (1 if fraction < schedule["near_turn"] else 2)

    def select(self, rng):
        stage = self.stage
        if self.config["curriculum"] == "gated" and stage and float(rng.random()) < self.gate["easy_mix"]:
            easier = list(range(stage))
            stage = easier[min(int(float(rng.random()) * len(easier)), len(easier) - 1)]
        choices = [x for x in self.layouts if x["layout_id"] == STAGES[stage]]
        index = min(int(float(rng.random()) * len(choices)), len(choices) - 1)
        return deepcopy(choices[index])

    def record(self, layout, success):
        key = _identity(layout)
        if key not in self.by_key:
            raise ValueError("课程只接受train布局，禁止val/test反馈")
        if type(success) is not bool:
            raise ValueError("success 必须是布尔值")
        item = self.by_key[key]
        if STAGES.index(item["layout_id"]) > self.stage:
            raise ValueError("不得记录尚未解锁的课程阶段")
        history = self.history[item["layout_id"]]
        history.append({"mirror": item["mirror"], "success": success})
        if self.config["curriculum"] != "gated" or self.stage == 2 or item["layout_id"] != self.stage_name:
            return
        if (len(history) >= self.gate["min_episodes"]
                and sum(x["success"] for x in history) / len(history) >= self.gate["success_rate"]
                and all(sum(x["success"] and x["mirror"] == side for x in history)
                        >= self.gate["min_side_success"] for side in ("left", "right"))):
            self.stage += 1

    def state_dict(self):
        return {"version": 1, "stage": self.stage, "game_tics": self.game_tics,
                "schedule": deepcopy(self.config["curriculum_schedule"]),
                "max_game_tics": self.config["max_game_tics"],
                "history": {s: list(deepcopy(self.history[s])) for s in STAGES},
                "mode": self.config["curriculum"], "gate": deepcopy(self.gate),
                "train_layouts": deepcopy(self.layouts)}

    def load_state_dict(self, state):
        _unknown(state, {"version", "stage", "game_tics", "schedule", "max_game_tics", "history", "mode", "gate", "train_layouts"}, "curriculum state")
        if state.get("max_game_tics") != self.config["max_game_tics"]:
            raise ValueError("课程恢复禁止改变max_game_tics以免静默重算fixed进度；可仅扩展wall_hours")
        if (state.get("version") != 1 or state.get("mode") != self.config["curriculum"]
                or state.get("gate") != self.gate or state.get("train_layouts") != self.layouts
                or state.get("schedule") != self.config["curriculum_schedule"]
                or state.get("max_game_tics") != self.config["max_game_tics"]):
            raise ValueError("课程state与配置不匹配")
        stage = state.get("stage")
        if type(stage) is not int or not 0 <= stage <= 2:
            raise ValueError("无效课程阶段")
        game_tics = state.get("game_tics")
        _number(game_tics, "game_tics", 0, integer=True)
        if state["mode"] == "fixed":
            fraction = game_tics / self.config["max_game_tics"]
            schedule = self.config["curriculum_schedule"]
            expected = 0 if fraction < schedule["exit"] else (1 if fraction < schedule["near_turn"] else 2)
            if stage != expected:
                raise ValueError("固定课程阶段与game_tics不一致")
        histories = state.get("history")
        if not isinstance(histories, dict) or set(histories) != set(STAGES):
            raise ValueError("无效课程历史")
        restored = {}
        for name, history in histories.items():
            if not isinstance(history, list) or len(history) > self.gate["window"]:
                raise ValueError("无效课程历史窗口")
            for item in history:
                if (not isinstance(item, dict) or set(item) != {"mirror", "success"}
                        or item["mirror"] not in ("left", "right") or type(item["success"]) is not bool):
                    raise ValueError("无效课程结果")
            restored[name] = deque(deepcopy(history), maxlen=self.gate["window"])
        self.stage, self.history, self.game_tics = stage, restored, game_tics
