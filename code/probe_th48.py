"""Probe readouts at scale=1.5, th=-48 (the combo that woke MN9 L+R)."""
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

def main():
    graph = ROOT / "outputs/doom/malecns_v1/graph.npz"
    manifest = json.loads((ROOT / "outputs/doom/malecns_v1/manifest.json").read_text())
    readouts = manifest["readouts"]
    brain = Brain(graph)
    brain.weight = (brain.weight * 1.5).astype(np.float32)
    game = Game(seed=41027, scenario="combat_survival")
    ro_spikes = {r["index"]: 0 for r in readouts}
    duration_ms = 50.0
    seconds = 5.0
    for k in range(int(seconds * 1000 / duration_ms)):
        if game.game.is_episode_finished():
            game.new_episode()
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain.uv)
        steps = int(round(duration_ms / brain.dt))
        alpha = 1 - np.exp(-steps * brain.dt / 10)
        brain.luminance += alpha * (np.clip(lum, 0, 1) - brain.luminance)
        brain.drive.fill(0)
        brain.drive[brain.lamina] = 12.0
        brain.drive[brain.retina] = 30 * brain.luminance / (0.02 + brain.luminance)
        brain.counts.fill(0)
        brain.cursor = advance_th(
            brain.ptr, brain.post, brain.weight, brain.v, brain.g, brain.refractory,
            brain.drive, brain.queue, brain.queue_count, brain.cursor, steps, brain.dt,
            brain.counts, brain.active, brain.active_flag, brain.nactive, -48.0)
        for r in readouts:
            c = int(brain.counts[r["index"]])
            if c:
                ro_spikes[r["index"]] += c
        game.act({"turn": 0, "forward": 0, "attack": 0})
    print(f"{'type':<9}{'side':<5}{'spikes/5s':>10}{'rateHz':>8}")
    for r in readouts:
        s = ro_spikes[r["index"]]
        print(f"{r['type']:<9}{r['side']:<5}{s:>10}{s/seconds:>8.1f}")
    (ROOT / "outputs/doom/readout-probe-th48.json").write_text(
        json.dumps({"scale": 1.5, "bias": 12.0, "thresh": -48.0, "seconds": seconds,
                    "spikes": {r["id"]: ro_spikes[r["index"]] for r in readouts}}, indent=2) + "\n")

if __name__ == "__main__":
    main()
