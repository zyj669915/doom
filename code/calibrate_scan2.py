"""Calibration scan 2: lamina_bias x weight_scale, normal vision vs blackout.

Goal: find a regime where biological readouts (MN9/DNa02/MDN/DNp09) fire WITH
vision but stay quiet when the screen is black (vision becomes the driver,
not the +12mV tonic lamina current).
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
OUT = ROOT / "outputs/doom/calibrate-scan2.json"
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
        ("bias0_scale4", 0.0, 4.0),
        ("bias0_scale8", 0.0, 8.0),
        ("bias2_scale4", 2.0, 4.0),
        ("bias4_scale4", 4.0, 4.0),
    ]
    seconds = 5.0
    duration_ms = 50.0
    n_steps = int(seconds * 1000 / duration_ms)
    results = []
    for name, bias, scale in combos:
        for cond in ["vision", "black"]:
            brain = Brain(GRAPH)
            brain.weight = (brain.weight * scale).astype(np.float32)
            game = Game(seed=41027, scenario="combat_survival")
            total_spikes = 0
            ro_spikes = {r["index"]: 0 for r in readouts}
            up_act = {r["index"]: 0 for r in readouts}
            up_spk = {r["index"]: 0 for r in readouts}
            t0 = time.perf_counter()
            for k in range(n_steps):
                if game.game.is_episode_finished():
                    game.new_episode()
                if cond == "black":
                    lum = np.zeros(len(brain.retina), dtype=np.float32)
                else:
                    rgb = game.pixels()
                    lum = retinal_samples(rgb, brain.uv)
                counts, _ = brain.step(lum, duration_ms, lamina_bias=bias)
                total_spikes += int(counts.sum())
                for r in readouts:
                    c = int(counts[r["index"]])
                    if c:
                        ro_spikes[r["index"]] += c
                for i, ups in up_ids.items():
                    tot = 0
                    act = 0
                    for u in ups:
                        c = int(counts[u])
                        if c:
                            tot += c
                            act += 1
                    up_spk[i] += tot
                    up_act[i] += act
                game.act({"turn": 0, "forward": 0, "attack": 0})
            elapsed = time.perf_counter() - t0
            def fmt(rows):
                return " ".join(f"{r['type']}{r['side'][0]}:{ro_spikes[r['index']]}/{up_act[r['index']]}"
                                for r in rows)
            rate = total_spikes / seconds
            results.append({"combo": name, "cond": cond, "bias": bias, "scale": scale,
                            "graph_rate_hz": round(rate, 1),
                            "bio": fmt(bio), "bci": fmt(bci),
                            "elapsed": round(elapsed, 1)})
            print(f"[{name}] {cond:6s}  graph={rate:9.1f}Hz  {elapsed:5.1f}s")
            print(f"   bio: {fmt(bio)}")
            print(f"   bci: {fmt(bci)}")
    OUT.write_text(json.dumps({"seconds": seconds, "results": results}, indent=2) + "\n")

if __name__ == "__main__":
    main()
