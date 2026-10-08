"""Quick probe: readout firing at the closed-loop regime (scale=1.5, bias=12)
for ALL 14 readouts, so we know which behavior channels are actually live."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from doom.engine import Brain
from doom.game import Game, retinal_samples

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
        counts, _ = brain.step(lum, duration_ms, lamina_bias=12.0)
        for r in readouts:
            c = int(counts[r["index"]])
            if c:
                ro_spikes[r["index"]] += c
        game.act({"turn": 0, "forward": 0, "attack": 0})
    print(f"{'type':<9}{'side':<5}{'spikes/5s':>10}{'rateHz':>8}")
    for r in readouts:
        s = ro_spikes[r["index"]]
        print(f"{r['type']:<9}{r['side']:<5}{s:>10}{s/seconds:>8.1f}")
    (ROOT / "outputs/doom/readout-probe-scale1.5.json").write_text(
        json.dumps({"scale": 1.5, "bias": 12.0, "seconds": seconds,
                    "spikes": {r["id"]: ro_spikes[r["index"]] for r in readouts}}, indent=2) + "\n")

if __name__ == "__main__":
    main()
