"""开发验收：极短真实全脑采集/优化/冻结评测，不用于性能结论。"""
import json
from pathlib import Path
from .runner import load_config, train, evaluate, json_write


def validate(config):
    results = []
    for algorithm in ("dqn", "ppo"):
        options = {"batch_size": 2, "learning_starts": 2} if algorithm == "dqn" else {
            "rollout_steps": 8, "epochs": 1, "minibatch_size": 4}
        c = load_config(config, {"algorithm": algorithm, "arm": "real", "max_env_steps": 12,
            "max_tics": 2, "calibration_steps": 12, "checkpoint_every": 6, "wall_hours": .1,
            "learner": options})
        result = train(c, purpose="integration-test")
        evaluation = evaluate(result["checkpoint"], "val", episodes=1)
        controls = []
        if algorithm == "dqn":
            for intervention, policy in (("black", "learned"), ("scramble", "learned"), ("none", "random"), ("none", "untrained")):
                control = evaluate(result["checkpoint"], "val", episodes=1, intervention=intervention, policy=policy)
                controls.append({"run": control["run"], "conditions": control["conditions"]})
        results.append({"algorithm": algorithm, "training": result, "controls": controls,
                        "evaluation": {k: v for k, v in evaluation.items() if k != "episodes"},
                        "note": "仅12转移优化用于生产调用链验收，成绩无意义，非正式训练"})
    return results
