"""Calibration scan 4: retina drive gain, bias=0, vision vs black.

Goal: make the lamina -> motor chain fire from VISION (photoreceptor drive)
alone, with no tonic bias, so black frames are silent. Tracks lamina and
retina activity explicitly.
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
OUT = ROOT / "outputs/doom/calibrate-scan4.json"
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"

def custom_step(brain, lum, duration_ms, bias, retina_gain, sugar=False):
    steps = int(round(duration_ms / brain.dt))
    alpha = 1 - np.exp(-steps * brain.dt / 10)
    brain.luminance += alpha * (np.clip(lum, 0, 1) - brain.luminance)
    brain.drive.fill(0)
    brain.drive[brain.lamina] = bias
    brain.drive[brain.retina] = retina_gain * 30 * brain.luminance / (0.02 + brain.luminance)
    if sugar:
        brain.drive[brain.sugar] = 30
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
    bci = [r for r in readouts if r["type"] in ("DNp20", "DNpe017")]
    lamina_ids = [int(x) for x in np.load(GRAPH)["lamina"]]
    retina_ids = [int(x) for x in np.load(GRAPH)["retina"]]

    combos = [
        ("g2_s2", 2.0, 2.0),
        ("g3_s2", 3.0, 2.0),
        ("g2_s4", 2.0, 4.0),
        ("g3_s4", 3.0, 4.0),
    ]
    seconds = 5.0
    duration_ms = 50.0
    n_steps = int(seconds * 1000 / duration_ms)
    results = []
    for name, gain, scale in combos:
        for cond in ["vision", "black"]:
            brain = Brain(GRAPH)
            brain.weight = (brain.weight * scale).astype(np.float32)
            game = Game(seed=41027, scenario="combat_survival")
            total_spikes = 0
            ro_spikes = {r["index"]: 0 for r in readouts}
            lam_spikes = 0
            ret_spikes = 0
            t0 = time.perf_counter()
            for k in range(n_steps):
                if game.game.is_episode_finished():
                    game.new_episode()
                if cond == "black":
                    lum = np.zeros(len(brain.retina), dtype=np.float32)
                else:
                    rgb = game.pixels()
                    lum = retinal_samples(rgb, brain.uv)
                counts = custom_step(brain, lum, duration_ms, bias=0.0, retina_gain=gain)
                total_spikes += int(counts.sum())
                lam_spikes += int(counts[lamina_ids].sum())
                ret_spikes += int(counts[retina_ids].sum())
                for r in readouts:
                    c = int(counts[r["index"]])
                    if c:
                        ro_spikes[r["index"]] += c
                game.act({"turn": 0, "forward": 0, "attack": 0})
            elapsed = time.perf_counter() - t0
            def fmt(rows):
                return " ".join(f"{r['type']}{r['side'][0]}:{ro_spikes[r['index']]}" for r in rows)
            rate = total_spikes / seconds
            results.append({"combo": name, "cond": cond, "retina_gain": gain, "scale": scale,
                            "bias": 0.0, "graph_rate_hz": round(rate, 1),
                            "retina_spikes": ret_spikes, "lamina_spikes": lam_spikes,
                            "bio": fmt(bio), "bci": fmt(bci),
                            "elapsed": round(elapsed, 1)})
            print(f"[{name}] {cond:6s}  graph={rate:9.1f}Hz ret={ret_spikes} lam={lam_spikes} {elapsed:5.1f}s")
            print(f"   bio: {fmt(bio)}")
            print(f"   bci: {fmt(bci)}")
    OUT.write_text(json.dumps({"seconds": seconds, "results": results}, indent=2) + "\n")

if __name__ == "__main__":
    main()
