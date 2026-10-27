"""Vision front-end comparison experiment (fixed frame sequence, fair A/B).

Records ONE fixed 200-frame gameplay sequence (seed 41027, combat_survival,
teacher-driven, same-frame aligned teacher actions). Then runs 6 vision
front-end variants on the SAME frames through the real fly connectome:
  v0_raw        raw luminance (current baseline)
  v1_agc        per-frame contrast normalization (lamina-style AGC)
  v2_motion     temporal luminance difference channel
  v3_dark       dark-adaptation gain (low-light sensitivity up)
  v4_agc_motion AGC + motion
  v5_full       dark + motion + AGC

Metrics per variant:
  A  modulation ratio  : bright/black total readout spikes (+1 guard)
  B  feature flux      : mean frame-to-frame |change| over 14 readouts
  C  feature variance  : mean over readouts of time-std
  D  teacher separability: linear softmax held-out acc on (feature->action),
       proxy for the distillation ceiling (chance = 1/12 ~= 8.3%)
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
from distill_fly import custom_step, readout_indexes, SCALE, BIAS, THRESH, DUR_MS
from teacher_ppo import Policy, index_to_action, downsample

OUT = CODE / 'outputs' / 'doom' / 'vision-exp'
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path(r'D:\fly\doomfly\outputs\doom\malecns_v1\manifest.json')
BRAIN = Path(r'D:\fly\doomfly\outputs\doom\malecns_v1\graph.npz')
N_FRAMES = 200
SEED = 41027


def agc(lum, eps=1e-4):
    sd = lum.std()
    if sd < eps:
        return lum.copy()
    return 0.5 + 0.5 * np.tanh((lum - lum.mean()) / (sd + eps) / 1.5)


def motion(lum, prev, k=1.2):
    if prev is None:
        return lum.copy()
    return np.clip(lum + k * (lum - prev), 0, 1)


def dark(lum, K=0.15, eps=1e-3):
    g = min(6.0, 1.0 + K / (lum.mean() + eps))
    return np.clip(lum * g, 0, 1)


def frontends():
    return {
        'v0_raw':       (lambda l, p: l.copy(), 'raw luminance'),
        'v1_agc':       (lambda l, p: agc(l), 'contrast AGC'),
        'v2_motion':    (lambda l, p: motion(l, p), 'motion diff k=1.2'),
        'v3_dark':      (lambda l, p: dark(l), 'dark-adapt gain'),
        'v4_agc_motion':(lambda l, p: agc(motion(l, p)), 'AGC+motion'),
        'v5_full':      (lambda l, p: agc(dark(motion(l, p))), 'dark+motion+AGC'),
    }


def record():
    teacher = Policy()
    ck = torch.load(CODE / 'outputs' / 'doom' / 'teacher-v1' / 'teacher.pt',
                    map_location='cpu', weights_only=True)
    teacher.load_state_dict(ck['policy']); teacher.eval()
    game = Game(seed=SEED, scenario='combat_survival')
    brain0 = Brain(BRAIN)
    lum_seq, act_seq = [], []
    for t in range(N_FRAMES):
        if game.game.is_episode_finished():
            game.new_episode()
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain0.uv)
        o = downsample(rgb).ravel()
        with torch.no_grad():
            a = int(teacher.act(torch.from_numpy(o).unsqueeze(0))[0].item())
        lum_seq.append(lum); act_seq.append(a)
        game.act(index_to_action(a))
    game.close()
    lum_seq = np.stack(lum_seq).astype(np.float32)
    act_seq = np.array(act_seq)
    np.savez(OUT / 'seq.npz', lum=lum_seq, acts=act_seq)
    print(f'recorded {N_FRAMES} frames | label dist: '
          f'{np.bincount(act_seq, minlength=12).tolist()}', flush=True)
    return lum_seq, act_seq


def log(msg):
    with open(OUT / 'progress.log', 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%H:%M:%S")} {msg}\n')


def run_variant(fe, lum_seq, black=False, name=''):
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    feats, tot = [], 0
    prev = None
    t_start = time.perf_counter()
    for i, l in enumerate(lum_seq):
        l2 = np.zeros_like(l) if black else fe(l, prev)
        prev = l
        counts = custom_step(brain, l2, DUR_MS, bias=BIAS, thresh=THRESH)
        feats.append(counts[ridx].astype(np.float32))
        tot += int(counts.sum())
        if i % 20 == 19:
            dt = time.perf_counter() - t_start
            log(f'{name} frame {i+1}/{len(lum_seq)} elapsed={dt:.1f}s '
                f'per_frame={(dt / (i + 1)) * 1000:.0f}ms')
    F = np.stack(feats)
    log(f'{name} done: total_readout_spikes={tot} frames={len(F)}')
    return F, tot


def sep_score(F, acts, seed=7):
    mu, sd = F.mean(0), F.std(0) + 1e-6
    Fz = ((F - mu) / sd).astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(F)); ntr = int(0.8 * len(F))
    Xtr = torch.tensor(Fz[idx[:ntr]]); Ytr = torch.tensor(acts[idx[:ntr]], dtype=torch.long)
    Xte = torch.tensor(Fz[idx[ntr:]]); Yte = torch.tensor(acts[idx[ntr:]], dtype=torch.long)
    W = torch.zeros(F.shape[1], 12, requires_grad=True)
    b = torch.zeros(12, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, Ytr)
        loss.backward(); opt.step()
    acc = ((Xte @ W + b).argmax(1) == Yte).float().mean().item()
    cs, cnt = 0.0, 0
    classes = np.unique(acts)
    for a1 in classes:
        c1 = Fz[acts == a1].mean(0)
        for a2 in classes:
            if a2 <= a1:
                continue
            cs += float(np.linalg.norm(c1 - Fz[acts == a2].mean(0))); cnt += 1
    return acc, (cs / cnt if cnt else 0.0)


def sep_score_windowed(F, acts, win=5, seed=7):
    """5-frame window features (70 dims) -> action linear separability.
    Window for frame t contains F[t-4..t] (current last), labels = acts[win-1:].
    """
    if len(F) <= win:
        return 0.0, 0.0
    n = len(F) - (win - 1)
    W = np.stack([F[i:i + n] for i in range(win)], axis=1)  # (n, win, 14)
    W = W[:, ::-1, :].reshape(n, -1).astype(np.float32)     # past..current order
    Y = acts[win - 1:]
    mu, sd = W.mean(0), W.std(0) + 1e-6
    Wz = ((W - mu) / sd).astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(W)); ntr = int(0.8 * len(W))
    Xtr = torch.tensor(Wz[idx[:ntr]]); Ytr = torch.tensor(Y[idx[:ntr]], dtype=torch.long)
    Xte = torch.tensor(Wz[idx[ntr:]]); Yte = torch.tensor(Y[idx[ntr:]], dtype=torch.long)
    Wt = torch.zeros(W.shape[1], 12, requires_grad=True)
    b = torch.zeros(12, requires_grad=True)
    opt = torch.optim.Adam([Wt, b], lr=0.03)
    for _ in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ Wt + b, Ytr)
        loss.backward(); opt.step()
    acc = ((Xte @ Wt + b).argmax(1) == Yte).float().mean().item()
    cs, cnt = 0.0, 0
    classes = np.unique(Y)
    for a1 in classes:
        c1 = Wz[Y == a1].mean(0)
        for a2 in classes:
            if a2 <= a1:
                continue
            cs += float(np.linalg.norm(c1 - Wz[Y == a2].mean(0))); cnt += 1
    return acc, (cs / cnt if cnt else 0.0)


def main():
    global ridx
    log(f'=== vision experiment start | python={sys.executable} | frames={N_FRAMES} | seed={SEED} ===')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    ridx = readout_indexes(manifest)

    seq_file = OUT / 'seq.npz'
    if seq_file.exists():
        d = np.load(seq_file)
        lum_seq, act_seq = d['lum'], d['acts']
        log(f'reused sequence {lum_seq.shape}')
    else:
        lum_seq, act_seq = record()

    fes = frontends()
    Fb, tot_b = run_variant(fes['v0_raw'][0], lum_seq, black=True, name='black')

    rows = []
    for name, (fe, desc) in fes.items():
        t0 = time.perf_counter()
        F, tot = run_variant(fe, lum_seq, name=name)
        np.save(OUT / f'feats-{name}.npy', F.astype(np.float32))
        mod = (tot + 1.0) / (tot_b + 1.0)
        flux = float(np.abs(np.diff(F, axis=0)).mean())
        var = float(F.std(0).mean())
        acc, cent = sep_score(F, act_seq)
        wacc, wcent = sep_score_windowed(F, act_seq)
        row = {'variant': name, 'desc': desc, 'modulation': round(mod, 3),
               'flux': round(flux, 3), 'feature_std': round(var, 3),
               'sep_acc_1f': round(acc, 3), 'centroid_sep_1f': round(cent, 3),
               'sep_acc_5w': round(wacc, 3), 'centroid_sep_5w': round(wcent, 3),
               'brain_spikes': int(tot), 'black_spikes': int(tot_b),
               'elapsed_s': round(time.perf_counter() - t0, 1)}
        rows.append(row)
        json.dump(rows, open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
        log(f'[{name:12s}] mod={mod:.3f} flux={flux:.3f} feat_std={var:.3f} '
            f'sep1f={acc:.3f} sep5w={wacc:.3f} cent5w={wcent:.3f} '
            f'({time.perf_counter()-t0:.0f}s)')
    log('=== summary (fixed 200-frame sequence, chance sep_acc=0.083) ===')
    for r in rows:
        log(f"{r['variant']:12s} mod={r['modulation']:.3f} flux={r['flux']:.3f} "
            f"std={r['feature_std']:.3f} sep1f={r['sep_acc_1f']:.3f} "
            f"sep5w={r['sep_acc_5w']:.3f} cent5w={r['centroid_sep_5w']:.3f}")
    log('=== vision experiment complete ===')


if __name__ == '__main__':
    main()
