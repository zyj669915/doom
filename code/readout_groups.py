# -*- coding: utf-8 -*-
"""Output-side functional groups + temporal coding (audit section 4):
  A vision          T4/T5/LC/LPLC/L1-3 + visual_projection/visual_centrifugal
  B descending/motor descending_neuron/vnc_motor + DN*/MN*/GF/MDN prefixes
  C navigation      cb_intrinsic (central complex + mushroom body pool)
Each group top32 by the readout score; separability across temporal codings
(single / concat-5 / mean-5 / trend-5 / concat-10), all OFFLINE on existing
counts from the readout experiment (v4 front-end, teacher-driven 200 frames).

Outputs:
  readout-groups/results.json   per-config x per-coding separability
  readout-groups/group_sets.npz selected indices per group
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from readout_experiment import (lin_softmax, action_F, rankpct, NEURONS, CATALOG,
                                MANIFEST, SEQ, WIN)

OUT = CODE / 'outputs' / 'doom' / 'readout-groups'
OUT.mkdir(parents=True, exist_ok=True)
RX_OUT = CODE / 'outputs' / 'doom' / 'readout-exp'
BRIGHT = RX_OUT / 'counts_bright.npy'
BLACK = RX_OUT / 'counts_black.npy'
SEL = RX_OUT / 'selected_sets.npz'


def window_coding(F, acts, win, mode='concat', seed=7, iters=700, lr=0.03):
    n = len(F) - (win - 1)
    if n < 10:
        return 0.0
    idx = np.arange(n)[:, None] + np.arange(win)[None, :]
    w = F[idx]  # (n, win, d)
    if mode == 'concat':
        W = w[:, ::-1, :].reshape(n, -1)
    elif mode == 'mean':
        W = w.mean(1)
    elif mode == 'trend':
        W = w[:, -1] - w[:, 0]
    else:
        raise ValueError(mode)
    Y = acts[win - 1:]
    mu, sd = W.mean(0), W.std(0) + 1e-6
    acc, cent = lin_softmax(((W - mu) / sd).astype(np.float32), Y, seed=seed, iters=iters, lr=lr)
    return round(float(acc), 3), round(float(cent), 3)


def main():
    import json as _json
    manifest = _json.loads(MANIFEST.read_text(encoding='utf-8'))
    author_idx = np.asarray([r['index'] for r in manifest['readouts']], dtype=np.int64)
    d = np.load(SEQ)
    acts = d['acts']
    cb = np.load(BRIGHT)
    ck = np.load(BLACK)
    bright = cb.astype(np.float32)
    black = ck.astype(np.float32)
    n = bright.shape[1]

    nodes = pd.read_feather(NEURONS)
    cell_type = nodes['cell_type'].astype(str)
    try:
        superclass = pd.read_feather(CATALOG).set_index('source_id').reindex(
            nodes['source_id']).reset_index()['superclass'].fillna('unassigned').astype(str)
    except Exception:
        superclass = nodes.get('superclass', pd.Series(['unassigned'] * len(nodes))).fillna('unassigned')

    A = cell_type.str.startswith(('T4', 'T5', 'LC', 'LPLC', 'L1', 'L2', 'L3')) | \
        superclass.isin(['visual_projection', 'visual_centrifugal'])
    B = superclass.isin(['descending_neuron', 'vnc_motor']) | \
        cell_type.str.startswith(('DN', 'MN', 'GF', 'MDN'))
    C = superclass.isin(['cb_intrinsic'])
    masks = {'A_vision': A.to_numpy(), 'B_desc_motor': B.to_numpy(), 'C_nav': C.to_numpy()}
    log = {k: int(v.sum()) for k, v in masks.items()}
    print('group pool sizes:', log, flush=True)

    bright_tot = bright.sum(0)
    black_tot = black.sum(0)
    mod = (bright_tot + 1.0) / (black_tot + 1.0)
    ts = bright.std(0)
    Fa = action_F(bright, acts)
    score = 0.4 * rankpct(mod) + 0.3 * rankpct(ts) + 0.3 * rankpct(Fa)

    # reuse existing global top32 (data-driven, from readout experiment)
    sel = np.load(SEL)
    top32_global = np.asarray(sel['top32_deb'], dtype=np.int64)

    configs = {'author14': author_idx}
    for k, m in masks.items():
        order = np.argsort(-score)
        cand = order[m[order]]
        configs[k] = cand[:32]
    configs['ABC96'] = np.unique(np.concatenate([configs['A_vision'][:32],
                                                 configs['B_desc_motor'][:32],
                                                 configs['C_nav'][:32]]))
    configs['top32_global'] = top32_global

    codings = [('1f', 1, 'concat'), ('5f_concat', 5, 'concat'), ('5f_mean', 5, 'mean'),
               ('5f_trend', 5, 'trend'), ('10f_concat', 10, 'concat')]

    rows = []
    for name, sel_i in configs.items():
        F = bright[:, sel_i]
        Fb = black[:, sel_i]
        Ff = np.clip(F - Fb, 0, None).astype(np.float32)   # debias (same as _deb)
        mod_set = float((bright_tot[sel_i].sum() + 1) / (black_tot[sel_i].sum() + 1))
        row = {'set': name, 'n': int(len(sel_i)), 'modulation': round(mod_set, 3)}
        for cname, win, mode in codings:
            acc, cent = window_coding(Ff, acts, win, mode)
            row[f'sep_{cname}'] = acc
            row[f'cent_{cname}'] = cent
        rows.append(row)
        json.dump(rows, open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
        print(f"[{name:10s}] n={len(sel_i):3d} " +
              ' '.join(f"{c[0]}={row[f'sep_{c[0]}']:.3f}" for c in codings), flush=True)

    np.savez(OUT / 'group_sets.npz', **{k: np.asarray(v).astype(np.int64) for k, v in configs.items()},
             author_idx=author_idx.astype(np.int64))
    # interpretability: top-10 picked per group (id + type + superclass)
    ids = nodes['source_id'].to_numpy()
    picks = {}
    for k in ['A_vision', 'B_desc_motor', 'C_nav']:
        picks[k] = [{'index': int(i), 'id': str(ids[i]), 'type': str(cell_type.iloc[i]),
                     'superclass': str(superclass.iloc[i]), 'score': round(float(score[i]), 3)}
                    for i in configs[k][:10]]
    json.dump(picks, open(OUT / 'top10_groups.json', 'w'), ensure_ascii=False, indent=1)
    print('=== readout groups complete ===', flush=True)


if __name__ == '__main__':
    main()
