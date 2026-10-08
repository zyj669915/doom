"""Probe v3 retina: luminance(0.5x) + temporal-difference(1.5x) drive.
Vision should light up readouts; black (dL=0) should stay quiet."""
import json
import math
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

def v3_step(brain, lum, duration_ms, bias, thresh, a_lum, a_diff):
    steps = int(round(duration_ms / brain.dt))
    alpha = 1 - np.exp(-steps * brain.dt / 10)
    prev = brain.luminance.copy()
    brain.luminance += alpha * (np.clip(lum, 0, 1) - brain.luminance)
    dL = np.abs(brain.luminance - prev)
    brain.drive.fill(0)
    brain.drive[brain.lamina] = bias
    lum_part = a_lum * 30 * brain.luminance / (0.02 + brain.luminance)
    diff_part = a_diff * 30 * dL / (0.02 + dL)
    brain.drive[brain.retina] = lum_part + diff_part
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
    watch = {"MN9", "DNp20", "DNpe017"}
    duration_ms = 50.0
    seconds = 5.0
    results = []
    for cond in ["vision", "black", "still"]:
        brain = Brain(graph)
        brain.weight = (brain.weight * 1.5).astype(np.float32)
        game = Game(seed=41027, scenario="combat_survival")
        ro_spikes = {r["index"]: 0 for r in readouts}
        total = 0
        for k in range(int(seconds * 1000 / duration_ms)):
            if game.game.is_episode_finished():
                game.new_episode()
            if cond == "black":
                lum = np.zeros(len(brain.retina), dtype=np.float32)
            else:
                rgb = game.pixels()
                lum = retinal_samples(rgb, brain.uv)
            if cond == "still":
                lum = np.full(len(lum), float(np.mean(lum)), dtype=np.float32)
            counts = v3_step(brain, lum, duration_ms, bias=12.0, thresh=-48.0,
                             a_lum=0.5, a_diff=1.5)
            total += int(counts.sum())
            for r in readouts:
                c = int(counts[r["index"]])
                if c:
                    ro_spikes[r["index"]] += c
            game.act({"turn": 0, "forward": 0, "attack": 0})
        parts = " ".join(
            f"{r['type']}{r['side'][0]}:{ro_spikes[r['index']]}" for r in readouts if r["type"] in watch)
        results.append({"cond": cond, "graph_total": int(total), "readouts": parts})
        print(f"[{cond:6s}] graph_total={total:9d}  {parts}")
    (ROOT / "outputs/doom/probe-v3.json").write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
