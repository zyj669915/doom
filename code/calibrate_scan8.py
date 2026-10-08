"""Scan 8: bias sweep at th=-48, scale=1.5/2.0, vision vs black.
Find the bias where vision/blackout readout difference is MAXIMAL while
biological readouts still fire under vision."""
import json
import math
import sys
import time
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

def custom_step(brain, lum, duration_ms, bias, thresh):
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
    graph = ROOT / "outputs/doom/malecns_v1/graph.npz"
    manifest = json.loads((ROOT / "outputs/doom/malecns_v1/manifest.json").read_text())
    readouts = manifest["readouts"]
    bio_idx = [r["index"] for r in readouts]
    duration_ms = 50.0
    seconds = 5.0
    results = []
    for bias in [3.0, 6.0, 9.0, 12.0]:
        for scale in [1.5, 2.0]:
            cond_spikes = {}
            for cond in ["vision", "black"]:
                brain = Brain(graph)
                brain.weight = (brain.weight * scale).astype(np.float32)
                game = Game(seed=41027, scenario="combat_survival")
                ro = {i: 0 for i in bio_idx}
                for k in range(int(seconds * 1000 / duration_ms)):
                    if game.game.is_episode_finished():
                        game.new_episode()
                    if cond == "black":
                        lum = np.zeros(len(brain.retina), dtype=np.float32)
                    else:
                        rgb = game.pixels()
                        lum = retinal_samples(rgb, brain.uv)
                    counts = custom_step(brain, lum, duration_ms, bias=bias, thresh=-48.0)
                    for i in bio_idx:
                        c = int(counts[i])
                        if c:
                            ro[i] += c
                    game.act({"turn": 0, "forward": 0, "attack": 0})
                cond_spikes[cond] = ro
            mn9v = sum(cond_spikes["vision"][r["index"]] for r in readouts if r["type"] == "MN9")
            mn9b = sum(cond_spikes["black"][r["index"]] for r in readouts if r["type"] == "MN9")
            dnp20v = sum(cond_spikes["vision"][r["index"]] for r in readouts if r["type"] == "DNp20")
            dnp20b = sum(cond_spikes["black"][r["index"]] for r in readouts if r["type"] == "DNp20")
            ratio_mn9 = (mn9v - mn9b) / max(1, mn9b)
            ratio_d = (dnp20v - dnp20b) / max(1, dnp20b)
            results.append({"bias": bias, "scale": scale,
                            "MN9_vision": mn9v, "MN9_black": mn9b, "MN9_delta%": round(100 * ratio_mn9, 1),
                            "DNp20_vision": dnp20v, "DNp20_black": dnp20b, "DNp20_delta%": round(100 * ratio_d, 1)})
            print(f"bias={bias:4.1f} scale={scale}  MN9: v={mn9v:4d} b={mn9b:4d} ({100*ratio_mn9:+5.1f}%)   "
                  f"DNp20: v={dnp20v:4d} b={dnp20b:4d} ({100*ratio_d:+5.1f}%)")
    (ROOT / "outputs/doom/calibrate-scan8.json").write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
