"""无标签开放环输入依赖性检查；不是 FlyVis 生理验证。"""
import time
import numpy as np
from .runner import Pipeline, new_run, provenance, json_write


def diagnose(c, frames=8):
    if c["arm"] not in ("real", "rewired") or not 2 <= frames <= 35:
        raise ValueError("diagnose 仅支持全脑臂，frames=2..35")
    run = new_run(c, "diagnose")
    json_write(run / "provenance.json", provenance(c))
    pipeline = Pipeline(c)
    brain = pipeline.brain
    before = brain.weight.copy()
    summaries, features = {}, {}
    start = time.perf_counter()
    for mode in ("black", "static", "flash", "left", "right", "scramble"):
        pipeline.reset()
        pipeline.vision.intervention = "black" if mode == "black" else "scramble" if mode == "scramble" else "none"
        total = np.zeros(brain.n, np.int64)
        history = []
        actual = 0.
        for tic in range(frames):
            x = np.arange(164)[None, :]
            phase = (-tic if mode == "left" else tic if mode in ("right", "scramble") else 0) * 4
            image = np.broadcast_to((((x + phase) // 16) % 2 * 255).astype(np.uint8), (120, 164))
            if mode == "flash":
                image = np.full_like(image, 255 if tic >= frames // 2 else 0)
            rgb = np.repeat(image[:, :, None], 3, axis=2)
            history.append(pipeline.raw(rgb))
            total += brain.counts
            actual += pipeline.last_diagnostic["actual_ms"]
        hz = [float(total[idx].mean() * 1000 / actual) if len(idx) else 0.
              for idx in pipeline.groups.group_indices]
        summaries[mode] = {"spikes": int(total.sum()), "active_neurons": int(np.count_nonzero(total)),
                           "actual_ms": actual, "group_mean_hz": hz,
                           "finite": bool(np.isfinite(history).all())}
        features[mode] = np.stack(history)
    differences = {mode: float(np.linalg.norm(features[mode] - features["black"])) for mode in features if mode != "black"}
    direction_distance = float(np.linalg.norm(features["left"] - features["right"]))
    unchanged = np.array_equal(before, brain.weight)
    result = {"run": str(run), "trained": False, "frames_per_condition": frames,
              "conditions": summaries, "feature_distance_from_black": differences,
              "left_right_distance": direction_distance, "weights_unchanged": unchanged,
              "groups": list(pipeline.groups.group_names), "wall_s": time.perf_counter()-start,
              "interpretation": "仅查有限性、输入依赖和背景活动；非方向选择性/生理有效性验收，黑屏仍可有偏置活动"}
    json_write(run / "result.json", result)
    if not unchanged:
        raise RuntimeError("诊断改变了权重")
    return result
