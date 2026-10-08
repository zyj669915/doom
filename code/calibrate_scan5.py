"""Calibration scan 5: keep lamina bias (carrier) at 12mV, scale=2, and raise
retina drive gain. The photoreceptor->lamina link is INHIBITORY (histamine),
so bright vision pushes lamina DOWN while blackout leaves it free. We want the
vision/blackout readout difference to become large, not vanish.

Tracks bio readouts and lamina activity under vision vs black.
"""
import json
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from doom.engine import Brain, advance
from doom.game import Game, retinal_samples

MANIFEST = ROOT / "outputs/doom/malecns_v1/manifest.json"
OUT = ROOT / "outputs/doom/calibrate-scan5.json"
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"

def custom_step(brain, lum, duration_ms, bias, retina_gain):
    steps = int(round(duration_ms / brain.dt))
    alpha = 1 - np.exp(-steps * brain.dt / 10)
    brain.luminance += alpha * (np.clip(lum, 0, 1) - brain.luminance)
    brain.drive.fill(0)
    brain.drive[brain.lamina] = bias
    brain.drive[brain.retina] = retina_gain * 30 * brain.luminance / (0.02 + brain.luminance)
    brain.counts.fill(0)
    brain.cursor = advance(
        brain.ptr, brain.post, brain.weight, brain.v, brain.g, brain.refractory,
        brain.drive, brain.queue, brain.queue_count, brain.cursor, steps, brain.dt,
        brain.counts, brain.active, brain.active_flag, brain.nactive)
    brain.sim_ms += steps * brain.dt
    return brain.counts.copy()

def main():
    g = np.load(GRAPH)
    ptr, ids = g["ptr"], g["ids"]
    n = len(ids)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    manifest = json.loads(MANIFEST.read_text())
    readouts = manifest["readouts"]
    bio = [r for r in readouts if r["type"] in ("MN9", "DNa02", "MDN", "DNp09")]
    lamina_ids = [int(x) for x in np.load(GRAPH)["lamina"]]

    combos = [("g1", 1.0), ("g2", 2.0), ("g3", 3.0), ("g4", 4.0)]
    seconds = 5.0
    duration_ms = 50.0
    n_steps = int(seconds * 1000 / duration_ms)
    results = []
    for name, gain in combos:
        for cond in ["vision", "black"]:
            brain = Brain(GRAPH)
            brain.weight = (brain.weight * 2.0).astype(np.float32)  # scale=2
            game = Game(seed=41027, scenario="combat_survival")
            total_spikes = 0
            ro_spikes = {r["index"]: 0 for r in readouts}
            lam_spikes = 0
            t0 = time.perf_counter()
            for k in range(n_steps):
                if game.game.is_episode_finished():
                    game.new_episode()
                if cond == "black":
                    lum = np.zeros(len(brain.retina), dtype=np.float32)
                else:
                    rgb = game.pixels()
                    lum = retinal_samples(rgb, brain.uv)
                counts = custom_step(brain, lum, duration_ms, bias=12.0, retina_gain=gain)
                total_spikes += int(counts.sum())
                lam_spikes += int(counts[lamina_ids].sum())
                for r in readouts:
                    c = int(counts[r["index"]])
                    if c:
                        ro_spikes[r["index"]] += c
                game.act({"turn": 0, "forward": 0, "attack": 0})
            elapsed = time.perf_counter() - t0
            def fmt(rows):
                return " ".join(f"{r['type']}{r['side'][0]}:{ro_spikes[r['index']]}" for r in rows)
            rate = total_spikes / seconds
            results.append({"combo": name, "cond": cond, "retina_gain": gain, "scale": 2.0,
                            "bias": 12.0, "graph_rate_hz": round(rate, 1),
                            "lamina_spikes": lam_spikes, "bio": fmt(bio),
                            "elapsed": round(elapsed, 1)})
            print(f"[{name}] {cond:6s}  graph={rate:9.1f}Hz lam={lam_spikes:8d} {elapsed:5.1f}s")
            print(f"   bio: {fmt(bio)}")
    OUT.write_text(json.dumps({"seconds": seconds, "results": results}, indent=2) + "\n")

if __name__ == "__main__":
    main()
