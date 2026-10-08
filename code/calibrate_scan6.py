"""Calibration scan 6: threshold scan. Copies the engine's numba advance into
this experiment with a parameterized threshold, so we can find whether a lower
spike threshold wakes DNa02R / DNp09 without exploding the graph.

Regime: bias=12, scale in {1.5, 2.0}, threshold in {-45,-48,-51}.
"""
import json
import math
import time
import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from doom.engine import Brain
from doom.game import Game, retinal_samples

@njit
def advance_th(ptr, post, weight, v, g, refractory, drive, queue, queue_count,
               cursor, steps, dt, counts, active, active_flag, nactive, thresh):
    av = math.exp(-dt / 20)
    ag = math.exp(-dt / 5)
    coupling = (av - ag) / 3
    delay_slots = queue.shape[0]
    for step in range(steps):
        slot = cursor % delay_slots
        for k in range(nactive[0]):
            i = active[k]
            if refractory[i] > 0:
                refractory[i] -= 1
            if refractory[i] == 0:
                v[i] = -52 + (v[i] + 52) * av + drive[i] * (1 - av) + g[i] * coupling
                g[i] *= ag
                if v[i] > thresh:
                    counts[i] += 1
                    future = (cursor + int(round(1.8 / dt))) % delay_slots
                    queue[future, queue_count[future]] = i
                    queue_count[future] += 1
        for q in range(queue_count[slot]):
            i = queue[slot, q]
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                if refractory[j] > 0:
                    continue
                g[j] += weight[e]
                if active_flag[j] == 0:
                    active_flag[j] = 1
                    active[nactive[0]] = j
                    nactive[0] += 1
        queue_count[slot] = 0
        future = (cursor + int(round(1.8 / dt))) % delay_slots
        for q in range(queue_count[future]):
            i = queue[future, q]
            v[i] = -52
            g[i] = 0
            refractory[i] = int(round(2.2 / dt))
        cursor += 1
    return cursor

def custom_step(brain, lum, duration_ms, bias, thresh, scale):
    steps = int(round(duration_ms / brain.dt))
    alpha = 1 - np.exp(-steps * brain.dt / 10)
    brain.luminance += alpha * (np.clip(lum, 0, 1) - brain.luminance)
    brain.drive.fill(0)
    brain.drive[brain.lamina] = bias
    brain.drive[brain.retina] = 30 * brain.luminance / (0.02 + brain.luminance)
    brain.counts.fill(0)
    brain.cursor = advance_th(
        brain.ptr, brain.post, brain.weight, brain.v, brain.g, brain.refractory,
        brain.drive, brain.queue, brain.queue_count, brain.cursor, steps, brain.dt,
        brain.counts, brain.active, brain.active_flag, brain.nactive, thresh)
    brain.sim_ms += steps * brain.dt
    return brain.counts.copy()

def main():
    g = np.load(ROOT / "outputs/doom/malecns_v1/graph.npz")
    ptr, ids = g["ptr"], g["ids"]
    n = len(ids)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    manifest = json.loads((ROOT / "outputs/doom/malecns_v1/manifest.json").read_text())
    readouts = manifest["readouts"]
    bio = [r for r in readouts if r["type"] in ("MN9", "DNa02", "MDN", "DNp09")]
    up_ids = {r["index"]: pre[ptr[r["index"]]:ptr[r["index"] + 1]].tolist() for r in readouts}

    combos = [
        ("th45_s1.5", -45.0, 1.5),
        ("th48_s1.5", -48.0, 1.5),
        ("th48_s2", -48.0, 2.0),
        ("th51_s1.5", -51.0, 1.5),
        ("th51_s2", -51.0, 2.0),
    ]
    seconds = 5.0
    duration_ms = 50.0
    n_steps = int(seconds * 1000 / duration_ms)
    results = []
    for name, thresh, scale in combos:
        brain = Brain(ROOT / "outputs/doom/malecns_v1/graph.npz")
        brain.weight = (brain.weight * scale).astype(np.float32)
        game = Game(seed=41027, scenario="combat_survival")
        total_spikes = 0
        ro_spikes = {r["index"]: 0 for r in readouts}
        up_act = {r["index"]: 0 for r in readouts}
        t0 = time.perf_counter()
        for k in range(n_steps):
            if game.game.is_episode_finished():
                game.new_episode()
            rgb = game.pixels()
            lum = retinal_samples(rgb, brain.uv)
            counts = custom_step(brain, lum, duration_ms, bias=12.0, thresh=thresh, scale=scale)
            total_spikes += int(counts.sum())
            for r in readouts:
                c = int(counts[r["index"]])
                if c:
                    ro_spikes[r["index"]] += c
            for i, ups in up_ids.items():
                for u in ups:
                    if counts[u] > 0:
                        up_act[i] += 1
            game.act({"turn": 0, "forward": 0, "attack": 0})
        elapsed = time.perf_counter() - t0
        def fmt(rows):
            return " ".join(f"{r['type']}{r['side'][0]}:{ro_spikes[r['index']]}" for r in rows)
        rate = total_spikes / seconds
        results.append({"combo": name, "thresh": thresh, "scale": scale,
                        "graph_rate_hz": round(rate, 1), "bio": fmt(bio),
                        "elapsed": round(elapsed, 1)})
        print(f"[{name}] th={thresh} scale={scale}  graph={rate:9.1f}Hz {elapsed:5.1f}s")
        print(f"   bio: {fmt(bio)}")
    (ROOT / "outputs/doom/calibrate-scan6.json").write_text(
        json.dumps({"seconds": seconds, "results": results}, indent=2) + "\n")

if __name__ == "__main__":
    main()
