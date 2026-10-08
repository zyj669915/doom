"""Distill v2: fix the one-frame misalignment + add 5-frame feature window.

v1 problems fixed here:
- alignment: teacher label and fly features now come from the SAME frame
- information: features = 5-frame window of 14 readouts -> 70 dims (temporal
  context: the brain now 'knows' what it was doing 4 ticks ago)
- evaluation: 600 tics (10 s) instead of the 210-tics truncation
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
from teacher_ppo import index_to_action, obs_vec
from distill_fly import custom_step, readout_indexes, SCALE, BIAS, THRESH, DUR_MS

OUT = ROOT/'outputs'/'doom'/'distill-v2'
OUT.mkdir(parents=True, exist_ok=True)
N_ACT = 12
WIN = 5
FEAT = 14 * WIN  # 70

class StudentV2(nn.Module):
    def __init__(self, feat=FEAT, n_act=N_ACT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat, 256), nn.Tanh(),
                                 nn.Linear(256, 128), nn.Tanh(),
                                 nn.Linear(128, n_act))
    def forward(self, x):
        return self.net(x)

def collect(teacher, n_episodes=8, max_tics=600, teacher_kind='rule'):
    manifest = json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
    ridx = readout_indexes(manifest)
    feats, acts = [], []
    for ep in range(n_episodes):
        brain = Brain(ROOT/'outputs/doom/malecns_v1/graph.npz')
        brain.weight = (brain.weight * SCALE).astype(np.float32)
        game = Game(seed=SEEDS[ep % 3], scenario='combat_survival')
        ring = np.zeros((WIN, len(ridx)), np.float32)
        done = False
        for t in range(max_tics):
            if game.game.is_episode_finished(): break
            rgb = game.pixels()                    # SAME frame for brain & teacher
            lum = retinal_samples(rgb, brain.uv)
            counts = custom_step(brain, lum, DUR_MS)
            ring = np.roll(ring, 1, 0); ring[0] = counts[ridx]
            o = obs_vec_from(rgb)
            if teacher_kind == 'rule':
                a = teacher.decide(rgb)
                a_idx = index_to_action_inv(a)
            else:
                with torch.no_grad():
                    a_idx = int(teacher.act(torch.from_numpy(o).unsqueeze(0))[0].item())
            feats.append(ring.ravel().copy()); acts.append(a_idx)
            if teacher_kind == 'rule':
                game.act(a)
            else:
                game.act(index_to_action(a_idx))
        game.close()
        print(f'ep {ep+1}/{n_episodes}: {len(feats)} frames', flush=True)
    X = np.stack(feats).astype(np.float32)
    Y = np.array(acts, dtype=np.int64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    np.savez(OUT/'distill-data.npz', X=(X-mu)/sd, Y=Y, mu=mu, sd=sd)
    print('collected', X.shape, '| label dist:', np.bincount(Y, minlength=N_ACT))
    return X, Y, mu, sd

def obs_vec_from(rgb):
    from teacher_ppo import downsample
    return downsample(rgb).ravel()

def index_to_action_inv(a):
    """Reverse index_to_action: action dict -> discrete index (for rule teacher labels)."""
    from teacher_ppo import TURNS, FWDS, ATTK
    t = np.argmin(np.abs(np.array(TURNS) - a['turn']))
    f = np.argmin(np.abs(np.array(FWDS) - a['forward']))
    at = 1 if a['attack'] else 0
    return (t * len(FWDS) + f) * len(ATTK) + at

def train_student(Xs, Y, epochs=80):
    torch.manual_seed(41027)
    student = StudentV2()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    Xt = torch.from_numpy(Xs); Yt = torch.from_numpy(Y)
    n = len(Y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        s = 0
        for b in range(0, n, 256):
            bi = perm[b:b+256]
            loss = nn.functional.cross_entropy(student(Xt[bi]), Yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item() * len(bi)
        if ep % 10 == 0:
            acc = (student(Xt).argmax(1) == Yt).float().mean().item()
            print(f'epoch {ep}: loss={s/n:.4f} acc={acc:.3f}', flush=True)
    torch.save({'student': student.state_dict(), 'mu': None, 'sd': None,
                'scale': SCALE, 'bias': BIAS, 'thresh': THRESH}, OUT/'student.pt')
    return student

def evaluate(configs, tics=600):
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
            ring = np.zeros((WIN, len(ridx)), np.float32)
            t0 = time.perf_counter()
            for t in range(tics):
                if game.game.is_episode_finished(): break
                rgb = game.pixels()
                lum = retinal_samples(rgb, brain.uv)
                counts = custom_step(brain, lum, DUR_MS, bias=bias, thresh=thresh)
                if name == 'baseline':
                    r20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNp20' and r.get('side')=='R'])
                    l20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNp20' and r.get('side')=='L'])
                    dnpe = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNpe017'])
                    act = {'turn': float(np.clip((r20-l20)*0.12,-6,6)),
                           'forward': float(np.clip(dnpe*0.4,0,20)), 'attack': dnpe>0}
                else:
                    ring = np.roll(ring, 1, 0); ring[0] = counts[ridx]
                    f = (ring.ravel().astype(np.float32) - student.mu) / student.sd
                    with torch.no_grad():
                        ai = int(student(torch.from_numpy(f).unsqueeze(0)).argmax(1).item())
                    act = index_to_action(ai)
                game.act(act)
            import vizdoom
            health = int(game.game.get_game_variable(vizdoom.GameVariable.HEALTH))
            kills = int(game.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
            out.append({'config': name, 'seed': seed, 'tics': t+1, 'kills': kills,
                        'health': health, 'elapsed': round(time.perf_counter()-t0,1)})
            print(f'[{name:10s}] seed={seed} tics={t+1} kills={kills} health={health} ({time.perf_counter()-t0:.0f}s)', flush=True)
            game.close()
    json.dump(out, open(OUT/'evaluation.json','w'), ensure_ascii=False, indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in out: agg[r['config']].append(r)
    print('\n=== summary ===')
    for cfg, rows in agg.items():
        print(f'{cfg:10s} kills={np.mean([r["kills"] for r in rows]):.2f} tics={np.mean([r["tics"] for r in rows]):.0f} health={np.mean([r["health"] for r in rows]):.1f}')

SEEDS = [41027, 41028, 41029]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='rule', choices=['rule', 'ppo'])
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--tics', type=int, default=600)
    ap.add_argument('--reuse-data', action='store_true')
    args = ap.parse_args()

    if args.teacher == 'ppo':
        from teacher_ppo import Policy
        teacher = Policy()
        ck = torch.load(ROOT/'outputs/doom/teacher-v1/teacher.pt', map_location='cpu', weights_only=True)
        teacher.load_state_dict(ck['policy']); teacher.eval()
        tkind = 'ppo'
    else:
        from rule_teacher_v2 import RuleTeacher
        teacher = RuleTeacher()
        tkind = 'rule'

    if args.reuse-data and (OUT/'distill-data.npz').exists():
        d = np.load(OUT/'distill-data.npz')
        Xs, Y, mu, sd = d['X'], d['Y'], d['mu'], d['sd']
        print('reused data', Xs.shape)
    else:
        Xs, Y, mu, sd = collect(teacher, args.episodes, args.tics, tkind)
    student = train_student(Xs, Y)
    student.mu, student.sd = mu.astype(np.float32), sd.astype(np.float32)
    random_s = StudentV2(); random_s.mu, random_s.sd = mu.astype(np.float32), sd.astype(np.float32)
    evaluate([('random', random_s), ('distilled', student)], tics=args.tics)

if __name__ == '__main__':
    main()
