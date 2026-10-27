"""预锁短程任务与三arm配置；复用旧校验原语，不改旧协议。"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from flyrl.environment import LayoutConfig, RewardConfig, corridor_wad_bytes
from flyrl_next.config import DEFAULTS as NEXT_DEFAULTS, _learner, _number, _unknown

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "malecns-near-turn-three-arm-v1"
DEFAULTS = deepcopy(NEXT_DEFAULTS)
DEFAULTS.update(protocol=PROTOCOL, arm="real", curriculum="single", route="A2",
                algorithm="dqn", max_game_tics=50000, wall_hours=12., max_tics=1200,
                calibration_steps=128, output="experiment/runs_stage_20261026",
                checkpoint_every=1000, ram_gib=1.5, commit_gib=2.,
                evaluation_points=[10000, 30000, 50000],
                learner={"replay_capacity": 5000})
for split, offset, count in (("train", 71000, 12), ("val", 72000, 12), ("test", 73000, 24)):
    DEFAULTS[split + "_layouts"] = [asdict(LayoutConfig(
        layout_id="near_turn", layout_seed=offset+i, mirror="left" if i % 2 == 0 else "right",
        spawn_jitter=32., angle_jitter=20., lightlevel=224)) for i in range(count)]


def load_config(path=None, overrides=None):
    supplied = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8-sig"))
    c = deepcopy(DEFAULTS)
    for source in (supplied, overrides or {}):
        _unknown(source, DEFAULTS, "stage config")
        for key, value in source.items():
            if key in ("learner", "rewards", "curriculum_gate", "curriculum_schedule"):
                c[key].update(deepcopy(value))
            else:
                c[key] = deepcopy(value)
    if c["protocol"] != PROTOCOL or c["schema_version"] != 1:
        raise ValueError("未知独立实验协议")
    if (c["route"], c["algorithm"], c["task"], c["curriculum"], c["device"]) != ("A2", "dqn", "corridor", "single", "cpu"):
        raise ValueError("本轮仅支持A2样式、CPU DoubleDQN、单阶段走廊")
    if c["arm"] not in ("real", "rewired", "frontend"):
        raise ValueError("arm必须为real/rewired/frontend")
    if c["frontend"] not in ("agc_motion", "raw", "adapt", "contrast"):
        raise ValueError("未知感觉前端")
    if c["objective"] != "potential" or c["action_set"] != "full":
        raise ValueError("本轮保持potential奖励与完整动作集")
    for key in ("seed", "rewire_seed", "projection_seed"):
        _number(c[key], key, 0, integer=True)
    for key in ("max_tics", "max_game_tics", "action_repeat", "checkpoint_every", "calibration_steps", "epsilon_decay_steps"):
        _number(c[key], key, 1, integer=True)
    for key in ("ram_gib", "commit_gib", "wall_hours", "scale", "max_turn"):
        _number(c[key], key, 1e-9)
    _number(c["bias"], "bias")
    _number(c["threshold"], "threshold", -52, -1e-9)
    for key in ("epsilon_end", "gamma_tic"):
        _number(c[key], key, 0, 1)
    if c["max_game_tics"] > 50000 or c["wall_hours"] > 12 or c["action_repeat"] != 4:
        raise ValueError("首轮限额50ktic/12小时，动作保持4tic")
    if c["ram_gib"] < 1.5 or c["commit_gib"] < max(2., c["ram_gib"]):
        raise ValueError("未实测前预算下限为RAM 1.5GiB、commit 2GiB，且commit不得小于RAM")
    for key in ("output", "graph", "neurons", "catalog", "annotations"):
        if not isinstance(c[key], str) or not c[key]:
            raise ValueError("路径必须非空")
    seeds, hashes = set(), set()
    for split in ("train", "val", "test"):
        values = c[split + "_seeds"]
        if (not isinstance(values, list) or not values or any(type(x) is not int or x < 0 for x in values)
                or len(values) != len(set(values)) or seeds.intersection(values)):
            raise ValueError("各split环境seed必须非空、互斥且无重复")
        seeds.update(values)
        layouts = c[split + "_layouts"]
        if not isinstance(layouts, list) or not layouts:
            raise ValueError("布局列表不能为空")
        normalized = []
        for item in layouts:
            _unknown(item, LayoutConfig.__dataclass_fields__, split)
            layout = LayoutConfig(**item)
            if layout.layout_id != "near_turn" or layout.lightlevel != 224:
                raise ValueError("主协议限定near_turn、同光照224")
            digest = hashlib.sha256(corridor_wad_bytes(layout)).hexdigest()
            if digest in hashes:
                raise ValueError("实际WAD跨split或同split重复")
            hashes.add(digest)
            normalized.append(asdict(layout))
        if {x["mirror"] for x in normalized} != {"left", "right"}:
            raise ValueError("每split须覆盖左右")
        c[split + "_layouts"] = normalized
    if c["calibration_steps"] < 2 * len(c["train_layouts"]):
        raise ValueError("校准须覆盖所有训练布局")
    points = c["evaluation_points"]
    if not isinstance(points, list) or any(type(x) is not int or x <= 0 or x % 4 for x in points) or points != sorted(set(points)):
        raise ValueError("评测时点须递增唯一且为4tic倍数")
    _unknown(c["rewards"], RewardConfig.__dataclass_fields__, "rewards")
    RewardConfig(**c["rewards"])
    _learner(c["learner"], "dqn")
    return c


class Curriculum:
    """仅从预锁训练布局采样；不消费验证结果，不切换阶段。"""
    def __init__(self, config):
        self.c = load_config(overrides=config)
        self.game_tics = 0

    def select(self, rng):
        return deepcopy(self.c["train_layouts"][int(rng.integers(len(self.c["train_layouts"])))])

    def advance(self, tics):
        self.game_tics += tics

    def record(self, layout, success):
        if layout not in self.c["train_layouts"] or type(success) is not bool:
            raise ValueError("仅允许训练布局反馈")

    def state_dict(self):
        return {"protocol": PROTOCOL, "game_tics": self.game_tics,
                "train_layouts": deepcopy(self.c["train_layouts"])}

    def load_state_dict(self, state):
        if state.get("protocol") != PROTOCOL or state.get("train_layouts") != self.c["train_layouts"]:
            raise ValueError("单阶段采样协议不匹配")
        _number(state["game_tics"], "game_tics", 0, integer=True)
        self.game_tics = state["game_tics"]
