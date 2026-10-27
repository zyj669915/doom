"""Phase 3 closed-loop validation: 600 tics x 3 seeds x (baseline/random/distilled-v2)
plus final teacher eval (600 tics x 3 seeds + argmax action diversity).

Outputs:
  phase3/evaluation.json   per-seed rows
  phase3/summary.json      mean/std aggregates
  phase3/compare.png       bar chart (kills / survival / health)
"""
import json, sys, time
from pathlib import Path
import argparse
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from distill_fly import custom_step, readout_indexes, SCALE, BIAS, THRESH, DUR_MS
from distill_v2 import StudentV2, WIN
from teacher_ppo import Policy, index_to_action, obs_vec
from doom.engine import Brain
from doom.game import Game, retinal_samples

BRAIN_PATH = Path(r'D:\fly\doomfly\outputs\doom\malecns_v1\graph.npz')
MANIFEST = Path(r'D:\fly\doomfly\outputs\doom\malecns_v1\manifest.json')
SEEDS = [41027, 41028, 41029]
OUT = CODE / 'outputs' / 'doom' / 'phase3'

def baseline_action(counts, manifest):
    r20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type'] == 'DNp20' and r.get('side') == 'R'])
    l20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type'] == 'DNp20' and r.get('side') == 'L'])
    dnpe = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type'] == 'DNpe017'])
    return {'turn': float(np.clip((r20 - l20) * 0.12, -6, 6)),
            'forward': float(np.clip(dnpe * 0.4, 0, 20)), 'attack': dnpe > 0}

def run_closed_loop(name, seed, tics, manifest, ridx, student=None):
    brain = Brain(BRAIN_PATH)
    if name != 'baseline':
        brain.weight = (brain.weight * SCALE).astype(np.float32)
        bias, thresh = BIAS, THRESH
    else:
        bias, thresh = 12.0, -45.0
    game = Game(seed=seed, scenario='combat_survival')
    ring = np.zeros((WIN, len(ridx)), np.float32)
    t0 = time.perf_counter()
    for t in range(tics):
        if game.game.is_episode_finished():
            break
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain.uv)
        counts = custom_step(brain, lum, DUR_MS, bias=bias, thresh=thresh)
        if name == 'baseline':
            act = baseline_action(counts, manifest)
        else:
            ring = np.roll(ring, 1, 0)
            ring[0] = counts[ridx]
            f = (ring.ravel().astype(np.float32) - student.mu) / student.sd
            with torch.no_grad():
                ai = int(student(torch.from_numpy(f).unsqueeze(0)).argmax(1).item())
            act = index_to_action(ai)
        game.act(act)
    import vizdoom
    health = int(game.game.get_game_variable(vizdoom.GameVariable.HEALTH))
    kills = int(game.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
    game.close()
    elapsed = time.perf_counter() - t0
    print(f'[{name:9s}] seed={seed} tics={t+1} kills={kills} health={health} ({elapsed:.0f}s)', flush=True)
    return {'config': name, 'seed': seed, 'tics': t + 1, 'kills': kills, 'health': health}

def run_teacher_eval(seed, tics, teacher_path):
    teacher = Policy()
    ck = torch.load(teacher_path, map_location='cpu', weights_only=True)
    teacher.load_state_dict(ck['policy'])
    teacher.eval()
    game = Game(seed=seed, scenario='combat_survival')
    counts = np.zeros(12, dtype=np.int64)
    t0 = time.perf_counter()
    for t in range(tics):
        if game.game.is_episode_finished():
            break
        o = obs_vec(game)
        with torch.no_grad():
            ai = int(teacher.act(torch.from_numpy(o).unsqueeze(0))[0].item())
        counts[ai] += 1
        game.act(index_to_action(ai))
    import vizdoom
    health = int(game.game.get_game_variable(vizdoom.GameVariable.HEALTH))
    kills = int(game.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
    game.close()
    dist = (counts / counts.sum() if counts.sum() else counts)
    print(f'[teacher  ] seed={seed} tics={t+1} kills={kills} health={health} ({time.perf_counter()-t0:.0f}s)', flush=True)
    return {'config': 'teacher', 'seed': seed, 'tics': t + 1, 'kills': kills, 'health': health,
            'action_dist': dist.astype(float).tolist()}

def aggregate(rows):
    agg = {}
    for r in rows:
        agg.setdefault(r['config'], []).append(r)
    out = {}
    for cfg, rs in agg.items():
        out[cfg] = {'kills_mean': float(np.mean([r['kills'] for r in rs])),
                    'kills_std': float(np.std([r['kills'] for r in rs])),
                    'tics_mean': float(np.mean([r['tics'] for r in rs])),
                    'tics_std': float(np.std([r['tics'] for r in rs])),
                    'health_mean': float(np.mean([r['health'] for r in rs])),
                    'health_std': float(np.std([r['health'] for r in rs]))}
    return out

def plot(rows, summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfgs = ['baseline', 'random', 'distilled']
    labels = {'baseline': 'baseline', 'random': 'random', 'distilled': 'distilled'}
    x = np.arange(len(cfgs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.2))
    metrics = [('kills_mean', 'kills_std', 'Kills / game'),
               ('tics_mean', 'tics_std', 'Survival (tics)'),
               ('health_mean', 'health_std', 'Health (end)')]
    colors = ['#c0392b', '#27ae60', '#2980b9']
    for i, (m, s, lab) in enumerate(metrics):
        vals = [summary.get(c, {}).get(m, 0) for c in cfgs]
        errs = [summary.get(c, {}).get(s, 0) for c in cfgs]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, label=lab, color=colors[i],
               capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in cfgs])
    ax.set_title('closed-loop 600 tics x 3 seeds: baseline vs random vs distilled-v2')
    ax.legend()
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tics', type=int, default=600)
    ap.add_argument('--skip-student', action='store_true')
    ap.add_argument('--teacher-path', default=str(CODE / 'outputs' / 'doom' / 'teacher-v1' / 'teacher.pt'))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    ridx = readout_indexes(manifest)

    rows = []

    # closed-loop: baseline / random / distilled
    student = None
    if not args.skip_student:
        ck = torch.load(CODE / 'outputs' / 'doom' / 'distill-v2' / 'student.pt', map_location='cpu', weights_only=True)
        student = StudentV2()
        student.load_state_dict(ck['student'])
        student.mu = np.asarray(ck['mu'], np.float32)
        student.sd = np.asarray(ck['sd'], np.float32)
        student.eval()
        random_s = StudentV2()
        random_s.mu, random_s.sd = student.mu.copy(), student.sd.copy()
        random_s.eval()
        for name in ['baseline', 'random', 'distilled']:
            st = random_s if name == 'random' else (student if name == 'distilled' else None)
            for s in SEEDS:
                rows.append(run_closed_loop(name, s, args.tics, manifest, ridx, st))

    # teacher final eval + action diversity
    for s in SEEDS:
        rows.append(run_teacher_eval(s, args.tics, args.teacher_path))

    (OUT / 'evaluation.json').write_text(json.dumps(rows, indent=1) + '\n')
    summary = aggregate(rows)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('\n=== summary (mean over 3 seeds) ===')
    for cfg, s in summary.items():
        print(f'{cfg:9s} kills={s["kills_mean"]:.2f}+/-{s["kills_std"]:.2f} tics={s["tics_mean"]:.0f} health={s["health_mean"]:.1f}')
    if not args.skip_student:
        plot(rows, summary, OUT / 'compare.png')
        print('chart saved:', OUT / 'compare.png')

if __name__ == '__main__':
    main()
