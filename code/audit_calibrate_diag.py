"""Dynamics diagnosis: why do biological readouts (MN9/DNa02/MDN/DNp09) stay silent?

Runs the real game loop (combat_survival) for N neural-seconds, samples readout
membrane voltage v / conductance g, and counts upstream firing rates for the
readouts' direct presynaptic inputs. Read-only; no graph mutation.
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
OUT = ROOT / "outputs/doom/calibrate-diag.json"

def main():
    g = np.load(ROOT / "outputs/doom/malecns_v1/graph.npz")
    ptr, post, weight, ids = g["ptr"], g["post"], g["weight"], g["ids"]
    n = len(ids)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    manifest = json.loads(MANIFEST.read_text())
    readouts = manifest["readouts"]

    # reverse adjacency (readout -> its presynaptic inputs)
    order = np.argsort(post, kind="stable")
    rptr = np.r_[0, np.cumsum(np.bincount(post, minlength=n))].astype(np.int64)
    rpost = pre[order]

    brain = Brain(ROOT / "outputs/doom/malecns_v1/graph.npz")
    game = Game(seed=41027, scenario="combat_survival")

    total_sim_ms = 0.0
    frame = 0
    samples = []          # per-sample readout v/g
    upstream_counts = {}  # neuron -> total spikes observed (accumulated)

    diag_neurons = set(r["index"] for r in readouts)
    for r in readouts:
        s, e = ptr[r["index"]], ptr[r["index"] + 1]
        up = pre[s:e].tolist()
        upstream_counts.update({int(u): 0 for u in up})
    # also record upstream v for the top-excitatory inputs
    upstream_v = {}

    duration_ms = 50.0
    seconds = 12.0
    n_steps = int(seconds * 1000 / duration_ms)
    t0 = time.perf_counter()
    for k in range(n_steps):
        if game.game.is_episode_finished():
            game.new_episode()
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain.uv)
        counts, _ = brain.step(lum, duration_ms, lamina_bias=12.0)
        total_sim_ms += duration_ms
        frame += 1
        # accumulate upstream spikes
        for u in upstream_counts:
            c = int(counts[u])
            if c:
                upstream_counts[u] += c
        if k % 20 == 0:  # every 1s neural time
            vv = {str(i): float(brain.v[i]) for i in diag_neurons}
            gg = {str(i): float(brain.g[i]) for i in diag_neurons}
            for i in diag_neurons:
                upstream_v.setdefault(int(i), []).append(float(brain.v[i]))
            samples.append({"sim_ms": total_sim_ms, "v": vv, "g": gg})
        game.act({"turn": 0, "forward": 0, "attack": 0})
    elapsed = time.perf_counter() - t0

    rows = []
    for r in readouts:
        i = r["index"]
        s, e = ptr[i], ptr[i + 1]
        iw = weight[s:e]
        up = pre[s:e]
        pos_w = float(iw[iw > 0].sum())
        neg_w = float(iw[iw < 0].sum())
        # upstream firing stats
        up_ids = [int(u) for u in up]
        up_spikes = [upstream_counts.get(u, 0) for u in up_ids]
        up_total = int(sum(up_spikes))
        up_active = int(sum(1 for s0 in up_spikes if s0 > 0))
        vmax = max((x for x in upstream_v.get(i, [-52.0])), default=-52.0)
        vmean = float(np.mean(upstream_v.get(i, [-52.0])))
        over_thr = float(np.mean([1.0 if x > -45.0 else 0.0 for x in upstream_v.get(i, [-52.0])]))
        # g snapshot max
        gmax = max(s0["g"][str(i)] for s0 in samples)
        rows.append({
            "id": r["id"], "type": r["type"], "side": r["side"],
            "in_edges": int(e - s), "exc_w": round(pos_w, 2), "inh_w": round(neg_w, 2),
            "upstream_neurons": len(up_ids), "upstream_active": up_active,
            "upstream_total_spikes": up_total,
            "v_max": round(vmax, 2), "v_mean": round(vmean, 2),
            "frac_v_above_thr": round(over_thr, 3), "g_max": round(gmax, 3),
        })

    summary = {
        "scenario": "combat_survival", "seed": 41027, "neural_seconds": seconds,
        "game_frames": frame, "elapsed_sec": round(elapsed, 1),
        "lamina_bias": 12.0, "readouts": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'type':<9}{'side':<5}{'in':>5}{'excW':>8}{'inhW':>8}{'upAct':>7}{'upSpk':>8}"
          f"{'vMax':>8}{'vMean':>8}{'gMax':>7}")
    for r in rows:
        print(f"{r['type']:<9}{r['side']:<5}{r['in_edges']:>5}{r['exc_w']:>8.1f}{r['inh_w']:>8.1f}"
              f"{r['upstream_active']:>7}{r['upstream_total_spikes']:>8}"
              f"{r['v_max']:>8.2f}{r['v_mean']:>8.2f}{r['g_max']:>7.2f}")
    print("report ->", OUT)

if __name__ == "__main__":
    main()
