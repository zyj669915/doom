"""Decoder A/B on the same fixed frame: biological-role vs engineered BCI.

Same protocol as audit_experiments fixed-frame tests (18 frame intervals at
35 Hz), 3 vision conditions. Shows whether the biological motor readouts can
drive the game at all and whether vision changes their output.
"""
import gc, hashlib, json, time
from pathlib import Path
import numpy as np
from doom.native import NativeBrain
from doom.game import Game, retinal_samples
from doom.engine import NeuralControls

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "outputs/doom/malecns_v1/graph.npz"
M = json.loads((PATH.parent / "manifest.json").read_text())
OUT = ROOT / "outputs/doom/decoder-ab.json"

def sha(a): return hashlib.sha256(a.tobytes()).hexdigest()

def main():
    game = Game(); frame = game.pixels(); game.close()
    results = []
    for mode in ["biological", "bci"]:
        for condition in ["intact", "blank_vision", "retina_disconnected"]:
            b = NativeBrain(PATH); d = NeuralControls(M["readouts"], mode=mode)
            light = retinal_samples(frame, b.uv)
            if condition == "blank_vision": light.fill(0)
            if condition == "retina_disconnected":
                for i in b.retina: b.weight[b.ptr[i]:b.ptr[i + 1]] = 0
            total = np.zeros(b.n, dtype=np.int64); start = time.monotonic()
            for tick in range(18):
                steps = round((tick + 1) * 10000 / 35) - b.cursor
                counts, _ = b.step(light, steps * .1)
                total += counts; a = d.decode(counts, steps * .1 / 1000)
            results.append({
                "mode": mode, "condition": condition, "simulation_ms": b.sim_ms,
                "action": a, "readouts": a["readouts"],
                "bio_motor_spikes": int(total[[r["index"] for r in M["readouts"] if r["type"] in ("MN9", "MDN", "DNa02", "DNp09")]].sum()),
                "wall_seconds": round(time.monotonic() - start, 2)})
            del b; gc.collect()
    (OUT).write_text(json.dumps(results, indent=2) + "\n")
    print(f"{'mode':<11}{'condition':<20}{'turn':>7}{'fwd':>7}{'atk':>6}{'bioSpk':>8}")
    for r in results:
        a = r["action"]
        print(f"{r['mode']:<11}{r['condition']:<20}{a['turn']:>7.2f}{a['forward']:>7.2f}{str(a['attack']):>6}{r['bio_motor_spikes']:>8}")
    print("report ->", OUT)

if __name__ == "__main__":
    main()
