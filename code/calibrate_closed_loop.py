"""Closed-loop A/B: the decisive test of the calibration.

Configs:
  fix_bio      scale=1.5, bias=12, decode: MN9(attack, real motor neuron),
               DNp20 R-L (turn, real descending), DNpe017 (forward, real descending)
  fix_black    same as fix_bio but with black frames (vision ablation)
  baseline     original: scale=1, bias=12, author BCI decode (DNp20/DNpe017)
  random_conn  scale=1.5, bias=12, same decode as fix_bio, but synapse signs
               scrambled (amplitude preserved) -> broken connectome control

3 seeds x 1 episode each (210 tics or death), combat_survival.
"""
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

class Decoder:
    def __init__(self, readouts, mode="v1"):
        self.mode = mode
        self.readouts = readouts
        by = {}
        for r in readouts:
            by.setdefault((r["type"], r.get("side")), []).append(r["index"])
        self.mn9 = by.get(("MN9", None)) or [r["index"] for r in readouts if r["type"] == "MN9"]
        self.dnp20_r = by.get(("DNp20", "R"), [])
        self.dnp20_l = by.get(("DNp20", "L"), [])
        self.dnpe = [r["index"] for r in readouts if r["type"] == "DNpe017"]

    def decode(self, counts):
        def sumrate(idx_list):
            return sum(int(counts[i]) for i in idx_list) if idx_list else 0
        turn = float(np.clip((sumrate(self.dnp20_r) - sumrate(self.dnp20_l)) * 0.12, -6, 6))
        mn9_rate = sumrate(self.mn9)
        if self.mode == "bci":  # original author BCI
            dnpe_rate = sumrate(self.dnpe)
            fwd = float(np.clip(dnpe_rate * 0.4, 0, 20))
            attack = dnpe_rate > 0
        elif self.mode == "v1":
            fwd = float(np.clip(sumrate(self.dnpe) * 0.4, 0, 20))
            attack = mn9_rate > 0
        else:  # v2: real motor neuron drives movement AND attack
            fwd = float(np.clip(mn9_rate * 0.3, 0, 20))
            attack = mn9_rate > 0
        return {"turn": turn, "forward": fwd, "attack": attack}

def main():
    manifest = json.loads((ROOT / "outputs/doom/malecns_v1/manifest.json").read_text())
    readouts = manifest["readouts"]
    dec = Decoder(readouts)
    graph_path = ROOT / "outputs/doom/malecns_v1/graph.npz"
    base = np.load(graph_path)
    seeds = [41027, 41028, 41029]
    tics = 210
    duration_ms = 50.0
    configs = [
        ("fix_v4", 1.5, 6.0, "vision", None, "v2", -48.0),
        ("fix_black_v4", 1.5, 6.0, "black", None, "v2", -48.0),
        ("baseline_v4", 1.0, 12.0, "vision", None, "bci", -45.0),
    ]
    out = []
    logf = ROOT / "outputs/doom/calibrate-closed-loop.jsonl"
    if logf.exists():
        logf.unlink()
    for name, scale, bias, cond, conn, mode, thresh in configs:
        for seed in seeds:
            brain = Brain(graph_path)
            brain.weight = (brain.weight * scale).astype(np.float32)
            if conn == "scramble":
                signs = np.random.RandomState(seed).choice([-1.0, 1.0], size=len(brain.weight))
                brain.weight = (np.abs(brain.weight) * signs).astype(np.float32)
            dec = Decoder(readouts, mode=mode)
            game = Game(seed=seed, scenario="combat_survival")
            kills = health = ticks = 0
            t0 = time.perf_counter()
            for t in range(tics):
                if game.game.is_episode_finished():
                    break
                if cond == "black":
                    lum = np.zeros(len(brain.retina), dtype=np.float32)
                else:
                    rgb = game.pixels()
                    lum = retinal_samples(rgb, brain.uv)
                counts = custom_step(brain, lum, duration_ms, bias=bias, thresh=thresh)
                act = dec.decode(counts)
                game.act(act)
                ticks = t + 1
            obs = game.game.get_state()
            health = int(game.game.get_game_variable(
                __import__("vizdoom").GameVariable.HEALTH))
            kills = int(game.game.get_game_variable(
                __import__("vizdoom").GameVariable.KILLCOUNT))
            elapsed = time.perf_counter() - t0
            row = {"config": name, "seed": seed, "scale": scale, "bias": bias,
                   "cond": cond, "thresh": thresh, "mode": mode, "tics": ticks,
                   "kills": kills, "health": health,
                   "elapsed": round(elapsed, 1)}
            out.append(row)
            with logf.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{name:12s}] seed={seed}  tics={ticks:3d} kills={kills} health={health} ({elapsed:.0f}s)")
            game.close()
            sys.stdout.flush()
    # summary
    import collections
    agg = collections.defaultdict(list)
    for r in out:
        agg[r["config"]].append(r)
    print("\n=== summary (mean kills / mean tics survived / mean health) ===")
    for cfg, rows in agg.items():
        mk = np.mean([r["kills"] for r in rows])
        mt = np.mean([r["tics"] for r in rows])
        mh = np.mean([r["health"] for r in rows])
        print(f"{cfg:12s}  kills={mk:.2f}  tics={mt:.1f}  health={mh:.1f}")
    (ROOT / "outputs/doom/calibrate-closed-loop.json").write_text(
        json.dumps(out, indent=2) + "\n")

if __name__ == "__main__":
    main()
