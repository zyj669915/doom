# -*- coding: utf-8 -*-
"""走廊学生 v4：epsilon 混合数据（治 BC 分布偏移）+ top512 5帧+差分 读出 + 闭环
（bug A/B/C/D/E/F/H/K/L 修复）。

修复记录（对照报告 §5.1 / §8.5）：
  * bug A：mu/sd 只在训练组【原始窗口特征】上拟合；
  * bug B：离线/在线窗口统一 FeatureWindow；窗口按 seed 分组构建，不跨组；
  * bug C/D：topK、统计只在训练组拟合；
  * bug E：起点段增强（首 40 帧 ×5）只作用于训练组帧，不污染选型组；
  * bug F：模型选择用 held-out 选型 seed，评测用独立 held-out seeds；
  * bug H：指标口径统一；bug K：闭环每 episode 重置脑+清窗口；
  * bug L：结果写 runs/<run_id>/ + provenance.json。

用法：python corridor_student_eps.py --train 1 --n-eps 6 --collect-eval 1
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import vizdoom as vzd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain  # noqa: E402
from doom.game import Game, retinal_samples  # noqa: E402
from readout_experiment import agc, motion, custom_step, SCALE, BIAS, THRESH, DUR_MS, BRAIN  # noqa: E402
from corridor_teacher import load_teacher, ObsBuilder, head_to_action  # noqa: E402
from corridor_common import (  # noqa: E402
    FeatureWindow, classification_metrics, fit_stats, frames_of_group,
    make_groups, normalize, reset_brain, run_id, select_topk, split_by_group,
    write_provenance,
)

OUT = CODE / 'outputs' / 'doom' / 'corridor-student-eps'
OUT.mkdir(parents=True, exist_ok=True)
WAD = CODE.parent.parent / 'doomfly' / 'doom' / 'scenarios' / 'corridor_avoid.wad'
TEACHER_PIXEL = CODE / 'outputs' / 'doom' / 'corridor-teacher-v6' / 'teacher-corridor-pixel-final.pt'

SEEDS = [41027, 41028, 41029, 41030]
TRAIN_SEEDS = [41027, 41028, 41029]
VAL_SEEDS = [41030]
EVAL_SEEDS = [41031, 41032]
N_PER_SEED = 500
EPS = 0.3
K, WIN, DIFF = 512, 5, True
NCLS = 3
MAX_CLOSED_TICS = 700
SEG, REP = 40, 5           # 起点段增强：每训练 seed 前 40 帧重复 5 倍
UV = np.load(BRAIN)['uv']


def collect_eps(teacher_path, seeds, n_per_seed, eps):
    """每 seed 前一半 greedy、后一半 epsilon 采样（随机动作+老师恢复）。"""
    policy, meta = load_teacher(teacher_path)
    use_angle = bool(meta.get('use_angle', True))
    lum_l, act_l = [], []
    rng = np.random.RandomState(7)
    for seed in seeds:
        game = Game(seed=seed, scenario='corridor_avoid')
        game.new_episode()
        ob = ObsBuilder(use_angle=use_angle)
        for i in range(n_per_seed):
            if game.game.is_episode_finished():
                game.new_episode()
                ob = ObsBuilder(use_angle=use_angle)
            rgb = game.pixels()
            lum = retinal_samples(rgb, UV).astype(np.float32)
            lum_l.append(lum)
            o = ob.step(game) if ob.prev is not None else ob.reset(game)
            use_rand = (i >= n_per_seed // 2) and (rng.rand() < eps)
            if use_rand:
                at, af = rng.randint(3), rng.randint(2)
            else:
                with torch.no_grad():
                    at, af, _, _ = policy.act(torch.from_numpy(o).float().unsqueeze(0), greedy=True)
            act_l.append(int(at) * 2 + int(af))
            game.act(head_to_action(at, af))
        game.close()
        print(f'[collect-eps seed={seed}] {n_per_seed} frames (eps={eps})', flush=True)
    return np.stack(lum_l), np.asarray(act_l)


def sim(lum):
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    T = len(lum)
    counts = np.zeros((T, brain.n), dtype=np.float32)
    prev = None
    t0 = time.perf_counter()
    for i, l in enumerate(lum):
        l2 = agc(motion(l, prev))
        prev = l
        c = custom_step(brain, l2, DUR_MS, bias=BIAS, thresh=THRESH)
        counts[i] = c.astype(np.float32)
        if (i + 1) % 500 == 0:
            print(f'  [sim] {i + 1}/{T} frames, {time.perf_counter() - t0:.1f}s', flush=True)
    return counts


class MLPWin(nn.Module):
    def __init__(self, d_in, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, NCLS))

    def forward(self, x):
        return self.net(x)


def train_linear3(Xtr, Ytr, iters=800, lr=0.05, reg=1e-4):
    Xtr = Xtr.astype(np.float64)
    ntr, d = Xtr.shape
    W = np.zeros((d, NCLS), np.float64)
    b = np.zeros(NCLS, np.float64)
    for _ in range(iters):
        logits = Xtr @ W + b
        ex = np.exp(logits - logits.max(1, keepdims=True))
        P = ex / ex.sum(1, keepdims=True)
        G = P.copy()
        G[np.arange(ntr), Ytr] -= 1.0
        W -= lr * (Xtr.T @ G / ntr + reg * W)
        b -= lr * G.sum(0) / ntr
    return W, b


def predict_linear3(W, b, X):
    return (X.astype(np.float64) @ W + b).argmax(1)


def train_mlp(F_tr, Y_tr, F_va, Y_va, d_in, seed=0, epochs=40):
    torch.manual_seed(seed)
    model = MLPWin(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(F_tr, dtype=torch.float32)
    Yt = torch.tensor(Y_tr, dtype=torch.long)
    Xv = torch.tensor(F_va, dtype=torch.float32)
    best, best_state = -1, None
    n = len(F_tr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            out = model(Xt[b])
            loss = lossf(out, Yt[b])
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(Xv).argmax(1).numpy()
            acc = classification_metrics(Y_va, pred, NCLS)['balanced_acc']
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def train_mlp_full(F_all, Y_all, d_in, seed=42, epochs=60):
    torch.manual_seed(seed)
    model = MLPWin(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(F_all, dtype=torch.float32)
    Yt = torch.tensor(Y_all, dtype=torch.long)
    n = len(F_all)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            out = model(Xt[b])
            loss = lossf(out, Yt[b])
            loss.backward()
            opt.step()
    return model


def predict_mlp(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


def build_grouped(counts, groups, win_obj):
    parts = []
    for g in np.unique(groups):
        m = groups == g
        parts.append(win_obj.build(counts[m]))
    return np.concatenate(parts, axis=0)


def _teacher_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(tag, seeds, teacher):
    lum_p, acts_p = OUT / f'lum-{tag}.npy', OUT / f'acts-{tag}.npy'
    info_p = OUT / f'teacher_info-{tag}.json'
    if lum_p.exists() and acts_p.exists() and info_p.exists():
        info = json.loads(info_p.read_text(encoding='utf-8'))
        if info.get('sha256') == _teacher_sha(teacher):
            print(f'=== 复用 {tag} 数据（老师一致） ===', flush=True)
            return np.load(lum_p), np.load(acts_p)
    if not Path(teacher).exists():
        raise SystemExit(f'纯像素老师不存在：{teacher}，请先训练（--use-angle 0）')
    lum, acts = collect_eps(teacher, seeds, N_PER_SEED, EPS)
    np.save(lum_p, lum)
    np.save(acts_p, acts)
    info_p.write_text(json.dumps(
        {'teacher': str(teacher), 'sha256': _teacher_sha(teacher),
         'created': time.strftime('%Y-%m-%d %H:%M:%S %z')},
        ensure_ascii=False, indent=1), encoding='utf-8')
    return lum, acts


def load_counts(lum, tag):
    p = OUT / f'counts-real-{tag}.npy'
    if p.exists():
        print(f'=== 复用 {tag} counts ===', flush=True)
        return np.load(p)
    print(f'=== 全脑仿真（{tag}，真实连接组） ===', flush=True)
    t0 = time.perf_counter()
    counts = sim(lum)
    np.save(p, counts)
    print(f'  sim {time.perf_counter() - t0:.1f}s', flush=True)
    return counts


def train(collect_eval=True, teacher=None):
    teacher = teacher or TEACHER_PIXEL
    rid = run_id('stu-eps')
    run_dir = OUT / 'runs' / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    lum, acts6 = load_data('train', SEEDS, teacher)
    acts3 = (acts6 // 2).astype(np.int64)
    dist6 = np.bincount(acts6, minlength=6).tolist()
    dist3 = np.bincount(acts3, minlength=3).tolist()
    print(f'eps-data: acts6={dist6} turn3={dist3}', flush=True)

    counts = load_counts(lum, 'train')
    groups = make_groups(len(counts), SEEDS, N_PER_SEED)
    tr = frames_of_group(groups, TRAIN_SEEDS)
    va = frames_of_group(groups, VAL_SEEDS)
    split_by_group(groups, TRAIN_SEEDS, EVAL_SEEDS, val_ids=VAL_SEEDS)  # 防呆（bug E）

    idx = select_topk(counts[tr], acts3[tr], K, ncls=NCLS)     # bug C
    w = FeatureWindow(idx, win=WIN, diff=DIFF)
    F = build_grouped(counts, groups, w)                        # bug B/J
    mu, sd = fit_stats(F[tr])                                   # bug D
    Fz = normalize(F, mu, sd)

    F_tr, F_va = Fz[tr], Fz[va]
    Y_tr, Y_va = acts3[tr], acts3[va]

    # 模型选择（bug E/F）：选型组不参与增强
    Wl, bl = train_linear3(F_tr, Y_tr)
    lin_va = classification_metrics(Y_va, predict_linear3(Wl, bl, F_va), NCLS)
    _, mlp_va = train_mlp(F_tr, Y_tr, F_va, Y_va, F_tr.shape[1], seed=0)
    use_mlp = mlp_va >= lin_va['balanced_acc'] - 0.01
    print(f'[select] linear va bal={lin_va["balanced_acc"]} mlp va bal={mlp_va:.4f} '
          f'-> chosen={ "mlp" if use_mlp else "linear" }', flush=True)

    # 最终训练池 = 训练组 + 选型组；起点段增强只作用于训练组帧（bug E）
    m_all = tr | va
    base_idx = np.nonzero(m_all)[0]
    aug = []
    nseg = N_PER_SEED
    for s in TRAIN_SEEDS:
        g0 = int(np.nonzero(groups == s)[0][0])
        for _ in range(REP):
            aug.extend(range(g0, g0 + SEG))
    train_idx = np.concatenate([base_idx, np.asarray(aug, dtype=np.int64)])
    F_all, Y_all = Fz[train_idx], acts3[train_idx]
    print(f'start-segment augmentation: base={len(base_idx)} + aug={len(aug)} '
          f'-> {len(train_idx)} samples (seg={SEG} rep={REP})', flush=True)

    if use_mlp:
        model = train_mlp_full(F_all, Y_all, F_all.shape[1])
        state = model.state_dict()
        kind = 'mlp'
    else:
        Wl, bl = train_linear3(F_all, Y_all, iters=1200)
        state = {'W': Wl.astype(np.float32), 'b': bl.astype(np.float32)}
        kind = 'linear'

    ck = {'kind': kind, 'state': state, 'topk': idx, 'mu': mu, 'sd': sd,
          'win': WIN, 'diff': DIFF, 'run_id': rid,
          'train_seeds': TRAIN_SEEDS, 'val_seeds': VAL_SEEDS,
          'teacher': str(teacher), 'teacher_sha256': _teacher_sha(teacher)}
    ck_path = run_dir / 'student-eps-final.pt'
    torch.save(ck, ck_path)
    write_provenance(run_dir, {'run_id': rid, 'script': 'corridor_student_eps.py',
                               'train_seeds': TRAIN_SEEDS, 'val_seeds': VAL_SEEDS,
                               'eval_seeds': EVAL_SEEDS, 'n_per_seed': N_PER_SEED,
                               'eps': EPS, 'topk': K, 'win': WIN, 'diff': DIFF,
                               'seg': SEG, 'rep': REP},
                     sources={'teacher': teacher, 'brain_graph': BRAIN,
                              'corridor_wad': WAD, 'script': Path(__file__)})
    (run_dir / 'train-results.json').write_text(json.dumps(
        {'run_id': rid, 'kind': kind, 'dim': F.shape[1], 'linear_val': lin_va,
         'mlp_val_balanced': round(float(mlp_va), 4)},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'train-eps: dim={F.shape[1]} chosen={kind}', flush=True)

    if collect_eval:
        lum_ev, acts6_ev = load_data('eval', EVAL_SEEDS, teacher)
        acts3_ev = (acts6_ev // 2).astype(np.int64)
        counts_ev = load_counts(lum_ev, 'eval')
        groups_ev = make_groups(len(counts_ev), EVAL_SEEDS, N_PER_SEED)
        F_ev = build_grouped(counts_ev, groups_ev, FeatureWindow(idx, win=WIN, diff=DIFF))
        F_evz = normalize(F_ev, mu, sd)
        if kind == 'mlp':
            pred = predict_mlp(model, F_evz)
        else:
            pred = predict_linear3(state['W'], state['b'], F_evz)
        ev_metrics = classification_metrics(acts3_ev, pred, NCLS)
        (run_dir / 'eval-results.json').write_text(json.dumps(
            {'run_id': rid, 'eval_seeds': EVAL_SEEDS, 'metrics': ev_metrics},
            ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'[held-out eval] balanced={ev_metrics["balanced_acc"]} '
              f'(chance {ev_metrics["balanced_chance"]}) '
              f'ordinary={ev_metrics["ordinary_acc"]} (maj {ev_metrics["majority_frac"]})', flush=True)

    (OUT / 'latest.json').write_text(json.dumps(
        {'run_id': rid, 'student_eps': str(ck_path),
         'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    return rid


def closed_loop(n_eps, max_tics=MAX_CLOSED_TICS, seed=41031):
    latest = json.loads((OUT / 'latest.json').read_text(encoding='utf-8'))
    ck_path = OUT / 'runs' / latest['run_id'] / 'student-eps-final.pt'
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    idx, mu, sd = ck['topk'], ck['mu'], ck['sd']
    kind = ck['kind']
    d = len(idx)
    w = FeatureWindow(idx, win=ck['win'], diff=ck['diff'])
    if kind == 'mlp':
        model = MLPWin(w.dim)
        model.load_state_dict(ck['state'])
        model.eval()
    else:
        Wl, bl = ck['state']['W'], ck['state']['b']

    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)

    rid = run_id('stu-eps-cl')
    run_dir = OUT / 'runs' / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    game = Game(seed=seed, scenario='corridor_avoid')
    n_exit, n_wall, total_tics = 0, 0, 0
    ep_info = []
    for ep in range(n_eps):
        game.new_episode()
        reset_brain(brain)      # bug K
        w.reset()               # bug B
        prev = None
        tics, hits, max_prog, exited = 0, 0, 0.0, False
        while tics < max_tics:
            if game.game.is_episode_finished():
                break
            rgb = game.pixels()
            lum = retinal_samples(rgb, UV).astype(np.float32)
            l2 = agc(motion(lum, prev))
            prev = lum
            c = custom_step(brain, l2, DUR_MS, bias=BIAS, thresh=THRESH)
            feat = w.step(c[idx].astype(np.float32))
            F = normalize(feat, mu, sd)          # bug A
            with torch.no_grad():
                if kind == 'mlp':
                    logits = model(torch.from_numpy(F).float().unsqueeze(0)).squeeze(0)
                    ti = int(logits.argmax().item())
                else:
                    lgl = F.astype(np.float64) @ Wl + bl
                    ti = int(np.argmax(lgl))
            x0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
            y0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
            game.act(head_to_action(ti, 1))
            x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
            y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
            if math.hypot(x1 - x0, y1 - y0) < 0.5:
                hits += 1
            if x1 >= 1792 and y1 >= 2048:
                exited = True
                break
            if y1 < 256:
                prog = 1.0 - ((1792 - x1) + (2048 - 256)) / 3456.0
            elif x1 < 1792:
                prog = 1.0 - ((2048 - y1) + (1792 - x1)) / 3456.0
            else:
                prog = 1.0 - (2048 - y1) / 3456.0
            max_prog = max(max_prog, prog)
            tics += 1
        n_exit += int(exited)
        total_tics += tics
        n_wall += hits
        ep_info.append({'ep': ep, 'exit': exited, 'tics': tics, 'hits': hits,
                        'max_prog': round(max_prog, 3)})
        print(f'  ep={ep} exit={exited} tics={tics} hits={hits} maxprog={max_prog:.3f}', flush=True)
    game.close()
    res = {'student_kind': f'{kind}-eps-win', 'K': K, 'win': WIN, 'diff': DIFF, 'eps': EPS,
           'n_eps': n_eps, 'eval_seed': seed,
           'exit_rate': round(n_exit / n_eps, 3), 'exit_count': n_exit,
           'tics_mean': round(total_tics / n_eps, 1),
           'wall_rate': round(n_wall / total_tics, 4) if total_tics else 0.0,
           'episodes': ep_info, 'run_id': rid,
           'reference': {'teacher_exit_rate': 1.0, 'teacher_tics_mean': 565.2,
                         'teacher_wall_rate': 0.0007}}
    (run_dir / 'closed-loop-results.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    write_provenance(run_dir, {'run_id': rid, 'script': 'corridor_student_eps.py::closed_loop',
                               'eval_seed': seed, 'n_eps': n_eps, 'max_tics': max_tics},
                     sources={'student_ckpt': ck_path, 'brain_graph': BRAIN,
                              'corridor_wad': WAD, 'script': Path(__file__)})
    (OUT / 'latest.json').write_text(json.dumps(
        {'run_id': rid, 'closed_loop': str(run_dir / 'closed-loop-results.json'),
         'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print('closed-loop:', {k: v for k, v in res.items() if k != 'episodes'}, flush=True)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=int, default=1)
    p.add_argument('--n-eps', type=int, default=6)
    p.add_argument('--max-tics', type=int, default=MAX_CLOSED_TICS)
    p.add_argument('--teacher', type=str, default=str(TEACHER_PIXEL))
    p.add_argument('--collect-eval', type=int, default=1)
    p.add_argument('--eval-seed', type=int, default=41031)
    args = p.parse_args()
    if args.train:
        train(collect_eval=bool(args.collect_eval), teacher=Path(args.teacher))
    closed_loop(args.n_eps, args.max_tics, seed=args.eval_seed)
