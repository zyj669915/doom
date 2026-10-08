"""Calibration scan: does global synaptic weight scaling activate the silent
motor chain (MN9/DNa02/MDN/DNp09 upstreams) without exploding the graph?

Runs the real combat_survival loop for N neural-seconds per scale factor.
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
OUT = ROOT / "outputs/doom/calibrate-scan.json"
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"

def main():
    g = np.load(GRAPH)
    ptr, post, ids = g["ptr"], g["post"], g["ids"]
    n = len(ids)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    manifest = json.loads(MANIFEST.read_text())
    readouts = manifest["readouts"]
    bio = [r for r in readouts if r["type"] in ("MN9", "DNa02", "MDN", "DNp09")]
    bci = [r for r in readouts if r["type"] in ("DNp20", "DNpe017")]

    scales = [1.0, 2.0, 4.0, 8.0]
    seconds = 6.0
    duration_ms = 50.0
    n_steps = int(seconds * 1000 / duration_ms)
    results = []
    for scale in scales:
        brain = Brain(GRAPH)
        brain.weight = (brain.weight * scale).astype(np.float32)
        game = Game(seed=41027, scenario="combat_survival")
        total_spikes = 0
        readout_spikes = {r["index"]: 0 for r in readouts}
        up_activity = {r["index"]: 0 for r in readouts}
        up_total = {r["index"]: 0 for r in readouts}
        for r in readouts:
            s, e = ptr[r["index"]], ptr[r["index"] + 1]
            for u in pre[s:e]:
                pass  # precompute below
        # precompute upstream id lists once
        up_ids = {r["index"]: pre[ptr[r["index"]]:ptr[r["index"] + 1]].tolist() for r in readouts}
        t0 = time.perf_counter()
        for k in range(n_steps):
            if game.game.is_episode_finished():
                game.new_episode()
            rgb = game.pixels()
            lum = retinal_samples(rgb, brain.uv)
            counts, _ = brain.step(lum, duration_ms, lamina_bias=12.0)
            total_spikes += int(counts.sum())
            for r in readouts:
                c = int(counts[r["index"]])
                if c:
                    readout_spikes[r["index"]] += c
            for i, ups in up_ids.items():
                tot = 0
                act = 0
                for u in ups:
                    c = int(counts[u])
                    if c:
                        tot += c
                        act += 1
                up_total[i] += tot
                up_activity[i] += act
            game.act({"turn": 0, "forward": 0, "attack": 0})
        elapsed = time.perf_counter() - t0
        def bio_stats():
            rows = []
            for r in bio:
                rows.append(f"{r['type']}{r['side'][0]}:spk={readout_spikes[r['index']]},"
                            f"upAct={up_activity[r['index']]},upSpk={up_total[r['index']]}")
            return " ".join(rows)
        def bci_stats():
            rows = []
            for r in bci:
                rows.append(f"{r['type']}{r['side'][0]}:spk={readout_spikes[r['index']]},"
                            f"upAct={up_activity[r['index']]},upSpk={up_total[r['index']]}")
            return " ".join(rows)
        rate = total_spikes / seconds
        results.append({"scale": scale, "total_spikes": int(total_spikes),
                        "total_rate_hz": round(rate, 1),
                        "bio": bio_stats(), "bci": bci_stats(),
                        "elapsed_sec": round(elapsed, 1)})
        print(f"scale={scale:>3}  graph_rate={rate:8.1f} Hz  elapsed={elapsed:5.1f}s")
        print(f"   bio: {bio_stats()}")
        print(f"   bci: {bci_stats()}")
    OUT.write_text(json.dumps({"scenario": "combat_survival", "seconds": seconds, "results": results}, indent=2) + "\n")

if __name__ == "__main__":
    main()
