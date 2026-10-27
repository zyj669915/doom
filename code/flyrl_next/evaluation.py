"""严格的 v2 评估比较；不同动作协议永不合并排名。"""
from copy import deepcopy
import json
import math
from pathlib import Path


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _read(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("result 必须为 JSON 对象或路径")
    value = deepcopy(value)
    if "provenance" in value:
        for key in ("sources", "inputs", "versions"):
            value.setdefault(key, value["provenance"].get(key))
    if "training_game_tics" in value:
        value.setdefault("training_steps", value["training_game_tics"])
    value.setdefault("calibration_frames", value.get("config", {}).get("calibration_steps"))
    if "intervention" in value and "policy" in value:
        value.setdefault("conditions", {"intervention": value["intervention"], "policy": value["policy"]})
    value.setdefault("checkpoint_manifest_sha256", value.get("checkpoint_sha256"))
    if value.get("quick", False):
        raise ValueError("quick 评估不能进入正式比较")
    required = {"config", "split", "conditions", "training_steps", "calibration_frames",
                "sources", "inputs", "versions", "episodes", "checkpoint"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"评估协议字段缺失: {sorted(missing)}")
    if not value["episodes"] or not isinstance(value["config"], dict):
        raise ValueError("评估 episodes/config 为空或无效")
    for key in ("sources", "inputs", "versions"):
        if not isinstance(value[key], dict) or not value[key]:
            raise ValueError(f"缺少可验证的 {key}")
    if value["split"] not in ("val", "test"):
        raise ValueError("比较只接受 val/test 完整评估")
    for key in ("training_steps", "calibration_frames"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"无效预算: {key}")
    for episode in value["episodes"]:
        if not isinstance(episode, dict) or not {"seed", "layout", "assets", "return", "tics"} <= episode.keys():
            raise ValueError("episode 缺少 seed/layout/assets/return/tics")
        for key in ("return", "tics"):
            if isinstance(episode[key], bool) or not isinstance(episode[key], (int, float)) or not math.isfinite(episode[key]):
                raise ValueError(f"episode {key} 不是有限数")
    _canonical(value)
    return value


def _action_protocol(row):
    c = row["config"]
    protocol = row.get("action_space", row.get("action_codec", c.get("action_space", c.get("action_codec"))))
    if protocol is None:
        protocol = row.get("protocol", {}).get("action_space")
    if protocol is None and {"task", "action_set", "action_repeat", "max_turn"} <= c.keys():
        protocol = {key: c[key] for key in ("task", "action_set", "action_repeat", "max_turn")}
    if protocol is None:
        raise ValueError("必须显式记录 action_space/action_codec；不能根据动作数推断动作含义")
    return protocol


def _signature(row, mode):
    c = deepcopy(row["config"])
    # 所有未列出的配置（含学习器、编码器、动力学、预算）都严格相同。
    ignored = {"output", "device"}
    if mode == "performance":
        ignored |= {"arm", "route", "seed", "rewire_seed", "action_space", "action_codec", "action_set", "action_repeat", "max_turn"}
    elif mode == "connectome":
        ignored |= {"arm", "rewire_seed"}
    elif mode == "intervention":
        ignored |= {"intervention"}
    if mode == "performance":
        ignored |= {"algorithm", "learner", "frontend", "scale", "bias", "threshold", "projection_seed", "curriculum", "curriculum_gate", "hidden", "rollout_steps", "epsilon_end", "epsilon_decay_steps", "checkpoint_every", "train_seeds", "train_layouts"}
    c = {k: v for k, v in c.items() if k not in ignored}
    inputs = deepcopy(row["inputs"])
    if mode == "performance":
        for key in ("graph", "neurons", "catalog", "annotations"):
            inputs.pop(key, None)
    conditions = deepcopy(row["conditions"])
    if mode == "intervention":
        conditions.pop("intervention", None)
        conditions.pop("policy", None)
    sig = {"config": c, "split": row["split"], "conditions": conditions,
           "training_steps": None if mode == "performance" else row["training_steps"], "calibration_frames": row["calibration_frames"],
           "sources": row["sources"], "inputs": inputs, "versions": row["versions"],
           "episodes": [{k: e[k] for k in ("seed", "layout", "assets")} for e in row["episodes"]]}
    protocol = deepcopy(row.get("protocol", {}))
    if mode == "performance":
        protocol.pop("action_space", None)
        protocol.pop("action_codec", None)
    sig["protocol"] = protocol
    if mode == "intervention":
        sig["checkpoint"] = row["checkpoint"]
        sig["checkpoint_manifest_sha256"] = row.get("checkpoint_manifest_sha256")
        if not sig["checkpoint_manifest_sha256"]:
            raise ValueError("干预比较需要 checkpoint_manifest_sha256")
    return _canonical(sig)


def _metrics(row):
    episodes = row["episodes"]
    metrics = {"n": len(episodes), "return_mean": sum(e["return"] for e in episodes) / len(episodes),
               "mean_tics": sum(e["tics"] for e in episodes) / len(episodes)}
    if row["config"].get("task") == "combat":
        # combat 不把 corridor success 或 progress 当作主要结论。
        for metric, aliases in {"kills_mean": ("kills", "killcount", "KILLCOUNT"),
                                "hits_mean": ("hits",),
                                "terminal_health_mean": ("health",),
                                "survival_tics_mean": ("survival_tics", "tics")}.items():
            values = []
            for e in episodes:
                telemetry = e.get("telemetry", {})
                value = next((source[key] for source in (e, telemetry) for key in aliases if key in source), None)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"combat 缺少有限指标 {metric}")
                values.append(value)
            metrics[metric] = sum(values) / len(values)
    else:
        if any(type(e.get("success")) is not bool for e in episodes):
            raise ValueError("corridor 缺少布尔 success")
        metrics["success_rate"] = sum(e["success"] for e in episodes) / len(episodes)
        progress = [e.get("max_progress") for e in episodes]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in progress):
            raise ValueError("corridor 缺少有限 max_progress")
        metrics["max_progress_mean"] = sum(progress) / len(episodes)
        successful = [e["tics"] for e in episodes if e["success"]]
        metrics["successful_tics_mean"] = sum(successful) / len(successful) if successful else None
    return metrics


def compare(results, mode="performance"):
    """接受 result.json 路径或字典，返回按动作协议隔离的 leaderboards。

    performance 允许不同训练 seed/arm；intervention 固定 checkpoint；
    connectome 固定训练 seed，仅允许 real/rewired 图对照。
    """
    if mode not in {"performance", "intervention", "connectome"}:
        raise ValueError("mode 必须为 performance/intervention/connectome")
    rows = [_read(item) for item in results]
    if len(rows) < 2:
        raise ValueError("至少需要两个评估结果")
    if mode == "performance" and any(_signature(row, mode) != _signature(rows[0], mode) for row in rows[1:]):
        raise ValueError("performance 任务、奖励、输入权限、预算上限或评估协议不一致")
    groups = {}
    for row in rows:
        groups.setdefault(_canonical(_action_protocol(row)), []).append(row)
    if mode != "performance" and len(groups) != 1:
        raise ValueError("干预/连接组比较拒绝混合动作协议")
    if mode == "connectome" and any(r["config"].get("arm") not in {"real", "rewired"} for r in rows):
        raise ValueError("connectome 只允许显式 arm=real/rewired 且同 route 的拓扑消融；A1/A2 是不同编码，不能替代连接组对照")
    boards = []
    for action, members in groups.items():
        if mode == "performance":
            # 纯视觉可没有脑文件；双方都有时，仍必须使用同一权威输入。
            known_inputs = {}
            for member in members:
                for key, value in member["inputs"].items():
                    if key in known_inputs and known_inputs[key] != value:
                        raise ValueError(f"共享输入来源不一致: {key}")
                    known_inputs[key] = value
        reference = _signature(members[0], mode)
        if any(_signature(r, mode) != reference for r in members[1:]):
            raise ValueError("评估协议不一致：任务、预算、编码、奖励、来源、版本或评估 episode 不可混比")
        runs = [{"arm": r["config"].get("arm"), "seed": r["config"].get("seed"),
                 "rewire_seed": r["config"].get("rewire_seed"), "checkpoint": r["checkpoint"],
                 "route": r["config"].get("route"), "conditions": r["conditions"],
                 "training_game_tics": r["training_steps"], "summary": _metrics(r)} for r in members]
        if members[0]["config"].get("task") == "corridor":
            # 失败局不能因提前死亡/快速超时获益；return 只旁列，不参与决胜。
            runs.sort(key=lambda r: (-r["summary"]["success_rate"],
                                     -r["summary"]["max_progress_mean"],
                                     r["summary"]["successful_tics_mean"] if r["summary"]["successful_tics_mean"] is not None else float("inf")))
            ranking = ["success_rate_desc", "max_progress_mean_desc", "successful_tics_mean_asc"]
        else:
            runs.sort(key=lambda r: r["summary"]["return_mean"], reverse=True)
            ranking = ["return_mean_desc"]
        boards.append({"action_space": json.loads(action), "runs": runs, "ranking": ranking,
                       "actual_training_budget_equal": len({r["training_steps"] for r in members}) == 1})
    return {"comparable": True, "mode": mode, "leaderboards": boards,
            "cross_action_comparable": len(boards) == 1,
            "claim": "不同动作协议独立分榜；单次成绩不能证明连接组优势，需多训练种子及多重连实例。"}
