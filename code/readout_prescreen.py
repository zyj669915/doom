# -*- coding: utf-8 -*-
"""Stage-2 readout prescreen: can the fly brain decode each teacher's actions
through candidate readout sets?  3 teachers x 3 readout sets x consistency.

Pipeline per teacher (highscore / stable / distillable):
  1. play combat_survival 150 frames with teacher policy (sampled actions)
  2. record retinal luminance sequence + action labels
  3. full-brain sim (v4_agc_motion front-end) -> per-frame counts
  4. decode actions from readout activity (single-frame + 5-frame window),
     held-out linear softmax accuracy (= brain-readout consistency)

Outputs: outputs/doom/readout-prescreen/results.json
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain
from doom.game import Game, retinal_samples
from readout_experiment import sim_full, sep_score, sep_score_windowed, readout_indexes, SCALE, BRAIN, MANIFEST
from teacher_ppo import Policy, ObsBuilder, index_to_action_12

OUT = CODE / 'outputs' / 'doom' / 'readout-prescreen'
OUT.mkdir(parents=True, exist_ok=True)
FINAL = CODE / 'outputs' / 'doom' / 'teacher-v4-final'
GROUPS = np.load(CODE / 'outputs' / 'doom' / 'readout-groups' / 'group_sets.npz')
SELECTED = np.load(CODE / 'outputs' / 'doom' / 'readout-exp' / 'selected_sets.npz')
N_FRAMES = 150
SEED = 41027

TEACHERS = {
    'highscore': FINAL / 'teacher-highscore.pt',
    'stable': FINAL / 'teacher-stable.pt',
    'distillable': FINAL / 'teacher-distillable.pt',
}

def collect(teacher_path, brain, n_frames=N_FRAMES):
    ck = torch.load(teacher_path, map_location='cpu', weights_only=True)
    cfg = ck.get('cfg', {})
    obs_mode = cfg.get('obs_mode', 'A')
    policy = Policy(obs_dim={'A': 24 * 32 * 2, 'B': 24 * 32 * 4, 'C': 24 * 32 * 3}[obs_mode],
                    action_mode='12')
    policy.load_state_dict(ck['policy'])
    policy.eval()
    game = Game(seed=SEED, scenario='combat_survival')
    game.new_episode()
    ob = ObsBuilder(mode=obs_mode, rand=0.0)
    lum_l, act_l = [], []
    for i in range(n_frames):
        if game.game.is_episode_finished():
            game.new_episode()
            ob = ObsBuilder(mode=obs_mode, rand=0.0)
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain.uv).astype(np.float32)
        lum_l.append(lum)
        o = ob.step(game) if i > 0 else ob.reset(game)
        with torch.no_grad():
            a = int(policy.act(torch.from_numpy(o).unsqueeze(0))[0].item())
        act_l.append(a)
        game.act(index_to_action_12(a))
    game.close()
    return np.stack(lum_l), np.asarray(act_l)

def balanced_acc(F, acts, seed=7):
    """Per-class recall averaged (balanced), same linear softmax split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(F)); ntr = int(0.8 * len(F))
    Xtr, Ytr = F[idx[:ntr]].astype(np.float64), acts[idx[:ntr]]
    Xte, Yte = F[idx[ntr:]].astype(np.float64), acts[idx[ntr:]]
    ncls = 12
    W = np.zeros((F.shape[1], ncls), np.float64); b = np.zeros(ncls, np.float64)
    for _ in range(600):
        logits = Xtr @ W + b
        ex = np.exp(logits - logits.max(1, keepdims=True))
        P = ex / ex.sum(1, keepdims=True)
        G = P.copy(); G[np.arange(ntr), Ytr] -= 1.0
        W -= 0.05 * (Xtr.T @ G / ntr + 1e-4 * W)
        b -= 0.05 * G.sum(0) / ntr
    pred = (Xte @ W + b).argmax(1)
    rec = np.zeros(ncls)
    for c in range(ncls):
        m = Yte == c
        if m.sum() == 0:
            rec[c] = np.nan
        else:
            rec[c] = (pred[m] == c).mean()
    return round(float(np.nanmean(rec)), 3)

def main():
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    author_idx = np.asarray(readout_indexes(manifest), dtype=np.int64)
    sets = {
        'author14': author_idx,
        'top32': SELECTED['top32_raw'].astype(np.int64),
        'ABC96': GROUPS['ABC96'].astype(np.int64),
    }
    results = {}
    for tname, tpath in TEACHERS.items():
        lum, acts = collect(tpath, brain)
        print(f'[{tname}] collected {len(lum)} frames, label dist '
              f'{np.bincount(acts, minlength=12).tolist()}', flush=True)
        counts, _ = sim_full(lum, name=f'prescreen-{tname}')
        np.save(OUT / f'counts-{tname}.npy', counts)
        maj = acts.mean()  # placeholder, computed below
        row = {}
        for sname, sel in sets.items():
            F = counts[:, sel].astype(np.float32)
            acc1, cent1 = sep_score(F, acts)
            acc5, cent5 = sep_score_windowed(F, acts)
            b1 = balanced_acc(F, acts)
            row[sname] = {'n': int(len(sel)), 'consistency_1f': round(float(acc1), 3),
                          'consistency_5w': round(float(acc5), 3),
                          'balanced_1f': round(float(b1), 3),
                          'centroid_5w': round(float(cent5), 3)}
            print(f'  [{sname:9s}] 1f={acc1:.3f} bal1f={b1:.3f} 5w={acc5:.3f} cent5w={cent5:.3f}', flush=True)
        row['label_baseline'] = {'majority_frac': round(float(np.bincount(acts, minlength=12).max() / len(acts)), 3)}
        results[tname] = row
    json.dump(results, open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
    print('saved', OUT / 'results.json')

if __name__ == '__main__':
    main()
