# -*- coding: utf-8 -*-
"""Readout-channel experiment: which readout set + debias treatment gives the
best distillation features for the fly-Doom student.

Fixed 200-frame sequence (seed 41027, combat_survival, teacher-driven) with the
CHOSEN vision front-end v4_agc_motion applied. Readout selection is POST-HOC:
it never changes brain dynamics, so only 2 full-brain sims are needed (bright +
black), then pure-numpy scoring / selection / metrics.

Sets compared:
  author14_raw / author14_deb   author's 14 DNs, raw vs black-debiased
  rnd32_deb                     random 32 (control for data-driven selection)
  top32_raw / top32_deb / top64_deb   data-driven by score
  bio32_deb / bio64_deb         biological-prior pool (T4/T5/LC/LPLC/GF/L1-3/
                                visual_projection/descending/motor/centrifugal)
  mix32_deb                     author14 + top18 from the rest
Metrics: single-frame & 5-frame-window linear separability vs teacher actions
(chance 1/12 ~= 8.3%), centroid separation, visual contribution
(bright-black)/black of the selected set (the L0 debias target).
"""
import sys, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain

SCALE, BIAS, THRESH = 1.5, 6.0, -48.0
DUR_MS = 50.0


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
    return [r['index'] for r in manifest['readouts']]

OUT = CODE / 'outputs' / 'doom' / 'readout-exp'
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path(r'D:\fly\doomfly\outputs\doom\malecns_v1\manifest.json')
BRAIN = Path(r'D:\fly\doomfly\outputs\doom\malecns_v1\graph.npz')
SEQ = CODE / 'outputs' / 'doom' / 'vision-exp' / 'seq.npz'
CATALOG = Path(r'D:\fly\experiment\data-reference\brain-snapshot-v0\normalized\catalog.feather')
NEURONS = Path(r'D:\fly\experiment\data-reference\brain-snapshot-v0\normalized\neurons.feather')

N_FRAMES = 200
WIN = 5


def log(msg):
    with open(OUT / 'progress.log', 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%H:%M:%S")} {msg}\n')


def agc(lum, eps=1e-4):
    sd = lum.std()
    if sd < eps:
        return lum.copy()
    return 0.5 + 0.5 * np.tanh((lum - lum.mean()) / (sd + eps) / 1.5)


def motion(lum, prev, k=1.2):
    if prev is None:
        return lum.copy()
    return np.clip(lum + k * (lum - prev), 0, 1)


def sim_full(seq, black=False, name=''):
    """Full-brain sim over the frame sequence; returns per-frame per-neuron
    spike counts (T, n) float32 and per-neuron totals."""
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    counts_all = np.zeros((len(seq), brain.n), dtype=np.float32)
    prev = None
    t0 = time.perf_counter()
    for i, l in enumerate(seq):
        l2 = np.zeros_like(l) if black else agc(motion(l, prev))
        prev = l
        counts = custom_step(brain, l2, DUR_MS, bias=BIAS, thresh=THRESH)
        counts_all[i] = counts.astype(np.float32)
        if i % 25 == 24:
            dt = time.perf_counter() - t0
            log(f'{name} frame {i+1}/{len(seq)} elapsed={dt:.1f}s '
                f'per_frame={(dt / (i + 1)) * 1000:.0f}ms')
    totals = counts_all.sum(0)
    log(f'{name} done: frames={len(seq)} total_spikes={totals.sum():.0f}')
    return counts_all, totals


def lin_softmax(Fz, Y, ncls=12, iters=600, lr=0.05, seed=7, reg=1e-4):
    """Pure-numpy linear softmax held-out accuracy (no torch dependency)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Fz)); ntr = int(0.8 * len(Fz))
    Xtr, Ytr = Fz[idx[:ntr]].astype(np.float64), Y[idx[:ntr]]
    Xte, Yte = Fz[idx[ntr:]].astype(np.float64), Y[idx[ntr:]]
    d = Fz.shape[1]
    W = np.zeros((d, ncls), np.float64); b = np.zeros(ncls, np.float64)
    for _ in range(iters):
        logits = Xtr @ W + b
        ex = np.exp(logits - logits.max(1, keepdims=True))
        P = ex / ex.sum(1, keepdims=True)
        G = P.copy()
        G[np.arange(ntr), Ytr] -= 1.0
        W -= lr * (Xtr.T @ G / ntr + reg * W)
        b -= lr * G.sum(0) / ntr
    acc = float(((Xte @ W + b).argmax(1) == Yte).mean())
    cs, cnt = 0.0, 0
    for a1 in np.unique(Y):
        c1 = Fz[Y == a1].mean(0)
        for a2 in np.unique(Y):
            if a2 <= a1:
                continue
            cs += float(np.linalg.norm(c1 - Fz[Y == a2].mean(0))); cnt += 1
    return acc, (cs / cnt if cnt else 0.0)


def sep_score(F, acts):
    mu, sd = F.mean(0), F.std(0) + 1e-6
    return lin_softmax(((F - mu) / sd).astype(np.float32), acts)


def sep_score_windowed(F, acts, win=WIN, seed=7):
    if len(F) <= win:
        return 0.0, 0.0
    n = len(F) - (win - 1)
    W = np.stack([F[i:i + n] for i in range(win)], axis=1)
    W = W[:, ::-1, :].reshape(n, -1).astype(np.float32)
    Y = acts[win - 1:]
    mu, sd = W.mean(0), W.std(0) + 1e-6
    acc, cent = lin_softmax(((W - mu) / sd).astype(np.float32), Y, seed=seed, iters=700, lr=0.03)
    return acc, cent


def rankpct(x):
    return (np.argsort(np.argsort(x)) + 1.0) / len(x)


def action_F(Y, acts):
    """Vectorized one-way ANOVA F per neuron (between/within variance across
    the 12 teacher action groups). Higher = more action-predictive."""
    n = Y.shape[0]
    onehot = (acts[:, None] == np.arange(12)).astype(np.float32)
    gsum = onehot.T @ Y                      # (12, d)
    gcnt = onehot.sum(0).clip(min=1e-6)      # (12,)
    Y2 = Y * Y
    gsum2 = onehot.T @ Y2
    ssw = gsum2.sum(0) - (gsum * gsum / gcnt[:, None]).sum(0)
    ssb = (gsum * gsum / gcnt[:, None]).sum(0) - (Y.sum(0) ** 2) / n
    ssw = np.where(ssw < 1e-12, np.nan, ssw)
    F = (ssb / 11.0) / (ssw / (n - 12.0))
    return np.nan_to_num(F, nan=0.0)


def build_sets(score, author_idx, bio_mask, rng):
    n = len(score)
    order = np.argsort(-score)
    top = lambda k: order[:k]
    rest = np.setdiff1d(order, author_idx, assume_unique=True)
    bio_order = np.argsort(-np.where(bio_mask, score, -1.0))
    sets = {
        'author14_raw': author_idx,
        'rnd32_deb': rng.choice(n, 32, replace=False),
        'top32_raw': top(32),
        'top32_deb': top(32),
        'top64_deb': top(64),
        'bio32_deb': bio_order[:32],
        'bio64_deb': bio_order[:64],
        'mix32_deb': np.unique(np.concatenate([author_idx, rest[:18]])),
    }
    return sets


def main():
    log(f'=== readout experiment start | frames={N_FRAMES} | win={WIN} | chance=0.083 ===')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    author_idx = np.asarray(readout_indexes(manifest))

    d = np.load(SEQ)
    lum_seq, act_seq = d['lum'], d['acts']
    log(f'sequence {lum_seq.shape} label dist {np.bincount(act_seq, minlength=12).tolist()}')

    cb_file = OUT / 'counts_bright.npy'
    ck_file = OUT / 'counts_black.npy'
    if cb_file.exists() and ck_file.exists():
        counts_bright = np.load(cb_file); counts_black = np.load(ck_file)
        log(f'reused full-brain counts bright {counts_bright.shape}')
    else:
        counts_bright, _ = sim_full(lum_seq, name='bright')
        counts_black, _ = sim_full(lum_seq, black=True, name='black')
        np.save(cb_file, counts_bright)
        np.save(ck_file, counts_black)

    n = counts_bright.shape[1]
    bright_tot = counts_bright.sum(0)
    black_tot = counts_black.sum(0)

    # per-neuron scoring
    mod = (bright_tot + 1.0) / (black_tot + 1.0)
    ts = counts_bright.std(0)
    Fa = action_F(counts_bright, act_seq)
    score = 0.4 * rankpct(mod) + 0.3 * rankpct(ts) + 0.3 * rankpct(Fa)

    # biological prior pool
    nodes = pd.read_feather(NEURONS)
    cell_type = nodes['cell_type'].astype(str)
    try:
        superclass = pd.read_feather(CATALOG).set_index('source_id').reindex(
            nodes['source_id']).reset_index()['superclass'].fillna('unassigned').astype(str)
    except Exception as e:
        superclass = nodes.get('superclass', pd.Series(['unassigned'] * len(nodes))).fillna('unassigned')
        log(f'catalog superclass fallback: {e}')
    prefixes = ('T4', 'T5', 'LC', 'LPLC', 'GF', 'MDN', 'L1', 'L2', 'L3')
    bio_mask = cell_type.str.startswith(prefixes) | superclass.isin(
        ['visual_projection', 'descending_neuron', 'vnc_motor', 'visual_centrifugal'])
    log(f'biological pool size: {int(bio_mask.sum())} / {n}')

    sets = build_sets(score, author_idx, bio_mask.to_numpy(), np.random.RandomState(11))

    rows = []
    for name, sel in sets.items():
        sel = np.asarray(sel)
        deb = 'deb' in name
        F = counts_bright[:, sel]
        Fb = counts_black[:, sel]
        if deb:
            Ff = np.clip(F - Fb, 0, None).astype(np.float32)
        else:
            Ff = F.astype(np.float32)
        mod_set = float((bright_tot[sel].sum() + 1) / (black_tot[sel].sum() + 1))
        vis_contrib = float((bright_tot[sel].sum() - black_tot[sel].sum()) / (black_tot[sel].sum() + 1.0))
        acc1, cent1 = sep_score(Ff, act_seq)
        acc5, cent5 = sep_score_windowed(Ff, act_seq)
        row = {'set': name, 'n': int(len(sel)),
               'modulation': round(mod_set, 3), 'vis_contrib': round(vis_contrib, 3),
               'sep_acc_1f': round(acc1, 3), 'centroid_1f': round(cent1, 3),
               'sep_acc_5w': round(acc5, 3), 'centroid_5w': round(cent5, 3)}
        rows.append(row)
        json.dump(rows, open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
        log(f"[{name:12s}] n={len(sel):2d} mod={mod_set:.3f} vis={vis_contrib:.2f} "
            f"sep1f={acc1:.3f} sep5w={acc5:.3f} cent5w={cent5:.3f}")

    # keep selected indices for the v3 pipeline
    np.savez(OUT / 'selected_sets.npz',
             **{k: np.asarray(v).astype(np.int64) for k, v in sets.items()},
             author_idx=author_idx.astype(np.int64))

    # interpretability: top-20 picked neurons (id + type)
    pick = np.asarray(sets['top32_deb'])[:20]
    ids = nodes['source_id'].to_numpy()
    picks = [{'index': int(i), 'id': str(ids[i]), 'type': str(cell_type.iloc[i]),
              'superclass': str(superclass.iloc[i]),
              'score': round(float(score[i]), 3)} for i in pick]
    json.dump(picks, open(OUT / 'top20.json', 'w'), ensure_ascii=False, indent=1)

    log('=== summary (chance sep=0.083; vis_contrib = (bright-black)/black) ===')
    for r in rows:
        log(f"{r['set']:12s} n={r['n']:2d} vis={r['vis_contrib']:.2f} "
            f"sep1f={r['sep_acc_1f']:.3f} sep5w={r['sep_acc_5w']:.3f} "
            f"cent5w={r['centroid_5w']:.3f}")
    log('=== readout experiment complete ===')


if __name__ == '__main__':
    main()
