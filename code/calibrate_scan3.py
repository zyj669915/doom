"""Calibration scan 3: light-adaptive lamina bias (bias = k*mean(luminance))
vs fixed bias, at scale 2/3, vision vs black.

Hypothesis: fixed 12mV is a non-visual driver (black == vision). Making bias
follow ambient luminance should silence the graph on black frames while still
activating the motor chain under real vision.
"""
import json
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from doom.engine import Brain
from doom.game import Game, retinal_samples

MANIFEST = ROOT / "outputs/doom/malecns_v1/manifest.json"
OUT = ROOT / "outputs/doom/calibrate-scan3.json"
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"

def main():
    g = np.load(GRAPH)
    ptr, ids = g["ptr"], g["ids"]
    n = len(ids)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    manifest = json.loads(MANIFEST.read_text())
    readouts = manifest["readouts"]
    bio = [r for r in readouts if r["type"] in ("MN9", "DNa02", "MDN", "DNp09")]
    bci = [r for r in readouts if r["type"] in ("DNp20", "DNpe017")]
    up_ids = {r["index"]: pre[ptr[r["index"]]:ptr[r["index"] + 1]].tolist() for r in readouts}

    combos = [
        ("dyn12_s2", "dynamic", 2.0),
        ("dyn12_s3", "dynamic", 3.0),
        ("fix12_s2", "fixed", 2.0),
        ("dyn8_s3", "dynamic8", 3.0),
    ]
    seconds = 5.0
    duration_ms = 50.0
    n_steps = int(seconds * 1000 / duration_ms)
    results = []
    for name, mode, scale in combos:
        for cond in ["vision", "black"]:
            brain = Brain(GRAPH)
            brain.weight = (brain.weight * scale).astype(np.float32)
            game = Game(seed=41027, scenario="combat_survival")
            total_spikes = 0
            ro_spikes = {r["index"]: 0 for r in readouts}
            t0 = time.perf_counter()
            for k in range(n_steps):
                if game.game.is_episode_finished():
                    game.new_episode()
                if cond == "black":
                    lum = np.zeros(len(brain.retina), dtype=np.float32)
                else:
                    rgb = game.pixels()
                    lum = retinal_samples(rgb, brain.uv)
                if mode == "dynamic":
                    bias = 12.0 * float(np.clip(np.mean(lum), 0, 1))
                elif mode == "dynamic8":
                    bias = 8.0 * float(np.clip(np.mean(lum), 0, 1))
                else:
                    bias = 12.0
                counts, _ = brain.step(lum, duration_ms, lamina_bias=bias)
                total_spikes += int(counts.sum())
                for r in readouts:
                    c = int(counts[r["index"]])
                    if c:
                        ro_spikes[r["index"]] += c
                game.act({"turn": 0, "forward": 0, "attack": 0})
            elapsed = time.perf_counter() - t0
            def fmt(rows):
                return " ".join(f"{r['type']}{r['side'][0]}:{ro_spikes[r['index']]}" for r in rows)
            rate = total_spikes / seconds
            results.append({"combo": name, "cond": cond, "bias_mode": mode, "scale": scale,
                            "graph_rate_hz": round(rate, 1),
                            "bio": fmt(bio), "bci": fmt(bci),
                            "elapsed": round(elapsed, 1)})
            print(f"[{name}] {cond:6s}  graph={rate:9.1f}Hz  {elapsed:5.1f}s")
            print(f"   bio: {fmt(bio)}")
            print(f"   bci: {fmt(bci)}")
    OUT.write_text(json.dumps({"seconds": seconds, "results": results}, indent=2) + "\n")

if __name__ == "__main__":
    main()
