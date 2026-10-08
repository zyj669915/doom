"""Distill a PPO teacher into the fly brain (doomfly-rl style).

Pipeline (mini version, runs on one CPU):
  1. Teacher policy (teacher-v1/teacher.pt) collects trajectories: for each frame,
     run the REAL fly connectome (scale=1.5, bias=6, thresh=-48) and record the
     14 readout-neuron spike counts + teacher's action.
  2. Train a small readout network (14 -> 128 -> 12) by cross-entropy to imitate
     the teacher from fly-brain features only.
  3. Evaluate closed-loop: baseline (author BCI) vs random readout vs distilled
     readout (3 seeds x 210 tics, combat_survival).

This mirrors doomfly-rl's "frozen connectome + learned gains/output" philosophy,
reduced to the readout layer for a local runnable demo.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from doom.engine import Brain
from doom.game import Game, retinal_samples
from teacher_ppo import Policy, index_to_action, obs_vec  # reuse policy

import math
from numba import njit

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

OUT = ROOT / 'outputs' / 'doom' / 'distill-v1'
OUT.mkdir(parents=True, exist_ok=True)
SCALE, BIAS, THRESH = 1.5, 6.0, -48.0
TICS, DUR_MS = 210, 50.0
SEEDS = [41027, 41028, 41029]
N_ACT = 12

def custom_step(brain, lum, duration_ms, bias=BIAS, thresh=THRESH):
    steps = int(round(duration_ms / brain.dt))
    alpha = 1 - np.exp(-steps * brain.dt / 10)
    brain.luminance += alpha * (np.clip(lum, 0, 1) - brain.luminance)
    brain.drive.fill(0)
    brain.drive[brain.lamina] = bias
    brain.drive[brain.retina] = 30 * brain.luminance / (0.02 + brain.luminance)
    brain.counts.fill(0)
    brain.cursor = advance_th(brain.ptr, brain.post, brain.weight, brain.v, brain.g,
        brain.refractory, brain.drive, brain.queue, brain.queue_count, brain.cursor,
        steps, brain.dt, brain.counts, brain.active, brain.active_flag,
        brain.nactive, thresh)
    brain.sim_ms += steps * brain.dt
    return brain.counts.copy()

def readout_indexes(manifest):
    by = []
    for r in manifest['readouts']:
        by.append(r['index'])
    return by

class Student(nn.Module):
    def __init__(self, feat=14, n_act=N_ACT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(),
                                 nn.Linear(128, 64), nn.Tanh(),
                                 nn.Linear(64, n_act))
    def forward(self, x):
        return self.net(x)

def collect(teacher, n_episodes=8):
    """Teacher plays; per frame we run the fly brain and record (features, action)."""
    manifest = json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
    ridx = readout_indexes(manifest)
    feats, acts = [], []
    for ep in range(n_episodes):
        brain = Brain(ROOT/'outputs/doom/malecns_v1/graph.npz')
        brain.weight = (brain.weight * SCALE).astype(np.float32)
        game = Game(seed=SEEDS[ep % 3], scenario='combat_survival')
        o = obs_vec(game)
        done = False
        for t in range(TICS):
            if game.game.is_episode_finished(): break
            rgb = game.pixels()
            lum = retinal_samples(rgb, brain.uv)
            counts = custom_step(brain, lum, DUR_MS)
            f = counts[ridx].astype(np.float32)
            # teacher action
            with torch.no_grad():
                a = int(teacher.act(torch.from_numpy(o).unsqueeze(0))[0].item())
            feats.append(f); acts.append(a)
            game.act(index_to_action(a))
            o = obs_vec(game)
        game.close()
        print(f'episode {ep+1}/{n_episodes}: collected {len(feats)} frames so far', flush=True)
    X = np.stack(feats).astype(np.float32)
    Y = np.array(acts)
    # standardize features
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    np.savez(OUT/'distill-data.npz', X=X, Y=Y, mu=mu, sd=sd)
    print('collected', X.shape, 'label dist:', np.bincount(Y, minlength=N_ACT), flush=True)
    return X, Y, mu, sd

def train_student(X, Y, mu, sd, epochs=60):
    torch.manual_seed(41027)
    student = Student()
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    Xt = torch.from_numpy((X - mu) / sd)
    Yt = torch.from_numpy(Y).long()
    n = len(Y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        loss_sum = 0
        for b in range(0, n, 256):
            bi = perm[b:b+256]
            logits = student(Xt[bi])
            loss = nn.functional.cross_entropy(logits, Yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(bi)
        acc = (student(Xt).argmax(1) == Yt).float().mean().item()
        if ep % 10 == 0:
            print(f'epoch {ep}: loss={loss_sum/n:.4f} acc={acc:.3f}', flush=True)
    torch.save({'student': student.state_dict(), 'mu': mu, 'sd': sd,
                'scale': SCALE, 'bias': BIAS, 'thresh': THRESH}, OUT/'student.pt')
    return student

def evaluate(configs):
    manifest = json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
    ridx = readout_indexes(manifest)
    out = []
    for name, student in configs:
        for seed in SEEDS:
            brain = Brain(ROOT/'outputs/doom/malecns_v1/graph.npz')
            if name != 'baseline':
                brain.weight = (brain.weight * SCALE).astype(np.float32)
                bias, thresh = BIAS, THRESH
            else:
                bias, thresh = 12.0, -45.0
            game = Game(seed=seed, scenario='combat_survival')
            kills = health = ticks = 0
            t0 = time.perf_counter()
            for t in range(TICS):
                if game.game.is_episode_finished(): break
                rgb = game.pixels()
                lum = retinal_samples(rgb, brain.uv)
                counts = custom_step(brain, lum, DUR_MS, bias=bias, thresh=thresh)
                if name == 'baseline':
                    # author BCI: turn from DNp20 R-L, fwd/attack from DNpe017
                    r20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNp20' and r.get('side')=='R'])
                    l20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNp20' and r.get('side')=='L'])
                    dnpe = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNpe017'])
                    turn = float(np.clip((r20-l20)*0.12, -6, 6))
                    act = {'turn': turn, 'forward': float(np.clip(dnpe*0.4,0,20)), 'attack': dnpe>0}
                else:
                    f = counts[ridx].astype(np.float32)
                    f = ((f - student.mu) / student.sd).astype(np.float32)
                    with torch.no_grad():
                        logits = student(torch.from_numpy(f).unsqueeze(0))
                        ai = int(logits.argmax(1).item())
                    act = index_to_action(ai)
                game.act(act)
                ticks = t + 1
            import vizdoom
            health = int(game.game.get_game_variable(vizdoom.GameVariable.HEALTH))
            kills = int(game.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
            elapsed = time.perf_counter()-t0
            out.append({'config': name, 'seed': seed, 'tics': ticks, 'kills': kills,
                        'health': health, 'elapsed': round(elapsed,1)})
            print(f'[{name:10s}] seed={seed} tics={ticks} kills={kills} health={health} ({elapsed:.0f}s)', flush=True)
            game.close()
    json.dump(out, open(OUT/'evaluation.json','w'), ensure_ascii=False, indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in out: agg[r['config']].append(r)
    print('\n=== summary (mean kills / tics / health) ===')
    for cfg, rows in agg.items():
        print(f'{cfg:10s} kills={np.mean([r["kills"] for r in rows]):.2f} tics={np.mean([r["tics"] for r in rows]):.1f} health={np.mean([r["health"] for r in rows]):.1f}')

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse-data', action='store_true',
                    help='load existing distill-data.npz instead of recollecting')
    args = ap.parse_args()
    teacher = Policy()
    ckpt = torch.load(OUT.parent/'teacher-v1'/'teacher.pt', map_location='cpu', weights_only=True)
    teacher.load_state_dict(ckpt['policy'])
    teacher.eval()
    if args.reuse_data and (OUT/'distill-data.npz').exists():
        d = np.load(OUT/'distill-data.npz')
        X, Y, mu, sd = d['X'], d['Y'], d['mu'], d['sd']
        print('reused collected data', X.shape)
    else:
        X, Y, mu, sd = collect(teacher, n_episodes=8)
    student = train_student(X, Y, mu, sd)
    random_student = Student()
    torch.manual_seed(0); random_student.load_state_dict(Student().state_dict())
    random_student.mu = mu; random_student.sd = sd
    student.mu = mu; student.sd = sd
    evaluate([('baseline', None), ('random', random_student), ('distilled', student)])

if __name__ == '__main__':
    main()
