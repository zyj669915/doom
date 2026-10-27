# -*- coding: utf-8 -*-
"""走廊任务三臂消融（脑 vs 无脑）——Windows 端（bug C/D/E/F/G/H/L 修复）。

三臂（同一批 500 帧、同一走廊老师、同一学生结构、同一评测）：
  A 真实脑：counts(real)[:, topK]          —— 冻结连接组的神经活动
  B 直接特征：感光值 PCA→K 维                —— 无脑对照（人工前端直接喂学生）
  C 打乱脑：weight-shuffle 后 counts(shuf)[:, topK] —— 破坏结构对照（保留度分布）

修复记录（对照报告 §5.1 / §8.5）：
  * bug C/D：topK 选择、PCA、z-score 统计一律只在【训练组】拟合；
  * bug E/F：按 seed 分组划分；评测用独立 held-out seeds（默认 41031/41032），
             不再用同批数据重洗 5 次冒充 5 个评测 seed；
  * bug G：每臂在各自训练组数据上独立选 topK（打乱脑不再套真实脑索引）；
  * bug H：balanced accuracy 对 1/C、ordinary accuracy 对多数类，附
             confusion matrix 与逐类 recall；
  * bug L：结果写入 runs/<run_id>/，附 provenance.json（老师/脑图/WAD/源码哈希）。

用法：
  python corridor_ablation.py                     # 全流程（含评测组采集+仿真）
  python corridor_ablation.py --collect-eval 0    # 复用已落盘数据
"""
import argparse
import json
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
    action_F, classification_metrics, fit_stats, frames_of_group, make_groups,
    normalize, rankpct, run_id, select_topk, split_by_group, write_provenance,
)

OUT = CODE / 'outputs' / 'doom' / 'corridor-ablation'
OUT.mkdir(parents=True, exist_ok=True)
TEACHER = CODE / 'outputs' / 'doom' / 'corridor-teacher-v6' / 'teacher-corridor-final.pt'
WAD = CODE.parent.parent / 'doomfly' / 'doom' / 'scenarios' / 'corridor_avoid.wad'

SEEDS = [41027, 41028, 41029, 41030]      # 训练组（含 1 个选型 seed）
TRAIN_SEEDS = [41027, 41028, 41029]       # 特征选择/统计/模型拟合
VAL_SEEDS = [41030]                       # 模型选择（早停）
EVAL_SEEDS = [41031, 41032]               # 独立 held-out 评测（bug F）
N_PER_SEED = 250
N_FRAMES = N_PER_SEED * len(SEEDS)
K = 32
SHUF_SEED = 42
MLP_EPOCHS = 40
MLP_LR = 1e-3
NCLS = 6
DEV = 'cpu'

UV = np.load(BRAIN)['uv']


# ---------- 数据采集（bug I：按老师 meta 决定是否带 ANGLE；bug L：记录来源） ----------
def collect(teacher_path, seeds, n_per_seed, tag='train'):
    policy, meta = load_teacher(teacher_path)
    use_angle = bool(meta.get('use_angle', True))
    lum_l, act_l = [], []
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
            with torch.no_grad():
                at, af, _, _ = policy.act(torch.from_numpy(o).float().unsqueeze(0), greedy=True)
            label = int(at.item()) * 2 + int(af.item())   # 6 类：turn_i*2+fwd_i
            act_l.append(label)
            game.act(head_to_action(at.item(), af.item()))
        game.close()
        print(f'[collect-{tag} seed={seed}] {n_per_seed} frames', flush=True)
    return np.stack(lum_l), np.asarray(act_l)


def _save_teacher_info(tag):
    import hashlib
    h = hashlib.sha256(Path(TEACHER).read_bytes()).hexdigest()
    (OUT / f'teacher_info-{tag}.json').write_text(
        json.dumps({'teacher': str(TEACHER), 'sha256': h,
                    'created': time.strftime('%Y-%m-%d %H:%M:%S %z')},
                   ensure_ascii=False, indent=1), encoding='utf-8')


def _load_data(tag, seeds, collect_eval):
    lum_p, acts_p = OUT / f'lum-{tag}.npy', OUT / f'acts-{tag}.npy'
    if lum_p.exists() and acts_p.exists() and (OUT / f'teacher_info-{tag}.json').exists():
        print(f'=== 复用 {tag} 数据 ===', flush=True)
        return np.load(lum_p), np.load(acts_p)
    if tag == 'eval' and not collect_eval:
        raise SystemExit('评测组数据缺失：请加 --collect-eval 1 先采集')
    lum, acts = collect(TEACHER, seeds, N_PER_SEED, tag=tag)
    np.save(lum_p, lum)
    np.save(acts_p, acts)
    _save_teacher_info(tag)
    return lum, acts


# ---------- 全脑仿真（可注入 weight 修改；同一打乱权重复用 train/eval） ----------
def _sim(lum, w_scaled, name=''):
    brain = Brain(BRAIN)
    brain.weight = w_scaled.astype(np.float32)
    T = len(lum)
    counts = np.zeros((T, brain.n), dtype=np.float32)
    prev = None
    t0 = time.perf_counter()
    for i, l in enumerate(lum):
        l2 = agc(motion(l, prev))
        prev = l
        c = custom_step(brain, l2, DUR_MS, bias=BIAS, thresh=THRESH)
        counts[i] = c.astype(np.float32)
        if (i + 1) % 100 == 0:
            print(f'  [{name}] {i + 1}/{T} frames, {time.perf_counter() - t0:.1f}s', flush=True)
    return counts


def _load_counts(lum, tag, arm, w_real_scaled=None, w_shuf_scaled=None):
    p = OUT / f'counts-{arm}-{tag}.npy'
    if p.exists():
        print(f'=== 复用 {arm}-{tag} counts ===', flush=True)
        return np.load(p)
    w = w_real_scaled if arm == 'real' else w_shuf_scaled
    counts = _sim(lum, w, name=f'{arm}-{tag}')
    np.save(p, counts)
    return counts


# ---------- 特征（每臂独立选择/拟合，只在训练组） ----------
def pca_fit(X, k):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k], X.mean(0)


def pca_apply(X, Vt, mean):
    return (X - mean) @ Vt.T


# ---------- 学生模型（6 类） ----------
class MLP(nn.Module):
    def __init__(self, d_in, ncls=NCLS, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, ncls))

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, Ytr, Xva, Yva, seed, d_in):
    torch.manual_seed(seed)
    model = MLP(d_in).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, best_state = -1, None
    n = len(Xtr)
    for ep in range(MLP_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, Yt[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(Xv).argmax(1).numpy()
            acc = classification_metrics(Yva, pred, NCLS)['balanced_acc']
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def train_linear(Xtr, Ytr, seed=0):
    rng = np.random.RandomState(seed)
    Xtr = Xtr.astype(np.float64)
    ntr, d = Xtr.shape
    W = np.zeros((d, NCLS), np.float64)
    b = np.zeros(NCLS, np.float64)
    for _ in range(800):
        logits = Xtr @ W + b
        ex = np.exp(logits - logits.max(1, keepdims=True))
        P = ex / ex.sum(1, keepdims=True)
        G = P.copy()
        G[np.arange(ntr), Ytr] -= 1.0
        W -= 0.05 * (Xtr.T @ G / ntr + 1e-4 * W)
        b -= 0.05 * G.sum(0) / ntr
    return W, b


def predict_linear(W, b, X):
    return (X.astype(np.float64) @ W + b).argmax(1)


def predict_mlp(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


def build_arm(name, cfg, F_tr, F_va, F_te, Y_tr, Y_va, Y_te, eval_groups, eval_seeds):
    """对单臂训练线性/MLP（tr 拟合、va 选型），在 held-out 评测组上报指标。"""
    W, b = train_linear(F_tr, Y_tr)
    mlp, _ = train_mlp(F_tr, Y_tr, F_va, Y_va, seed=0, d_in=F_tr.shape[1])
    lin_pred = predict_linear(W, b, F_te)
    mlp_pred = predict_mlp(mlp, F_te)
    res = {
        'linear': classification_metrics(Y_te, lin_pred, NCLS),
        'mlp': classification_metrics(Y_te, mlp_pred, NCLS),
        'train_frames': int(len(F_tr)),
        'val_frames': int(len(F_va)),
        'test_frames': int(len(F_te)),
        'test_seeds': [int(s) for s in eval_seeds],
    }
    print(f'[{name}] linear bal={res["linear"]["balanced_acc"]} (chance {res["linear"]["balanced_chance"]}) '
          f'ord={res["linear"]["ordinary_acc"]} (maj {res["linear"]["majority_frac"]}) | '
          f'mlp bal={res["mlp"]["balanced_acc"]} ord={res["mlp"]["ordinary_acc"]}', flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--collect-eval', type=int, default=1, help='采集并仿真评测组（held-out seeds）')
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = p.parse_args()

    rid = run_id('abl')
    run_dir = OUT / 'runs' / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    # 固定打乱权重：train/eval 用同一份打乱（保持对照一致）
    brain0 = Brain(BRAIN)
    w_real = (brain0.weight * SCALE).astype(np.float32)
    w_shuf = (np.random.RandomState(SHUF_SEED).permutation(brain0.weight.copy()) * SCALE).astype(np.float32)
    del brain0

    lum, acts = _load_data('train', args.seeds, collect_eval=1)
    maj = float(np.bincount(acts, minlength=NCLS).max() / len(acts))
    print(f'[train] lum {lum.shape} label dist {np.bincount(acts, minlength=NCLS).tolist()} maj={maj:.3f}', flush=True)

    if args.collect_eval:
        lum_ev, acts_ev = _load_data('eval', EVAL_SEEDS, collect_eval=1)
        print(f'[eval ] lum {lum_ev.shape} dist {np.bincount(acts_ev, minlength=NCLS).tolist()}', flush=True)
    else:
        lum_ev, acts_ev = np.load(OUT / 'lum-eval.npy'), np.load(OUT / 'acts-eval.npy')

    print('=== 仿真：真实脑 / 打乱脑（训练组） ===', flush=True)
    counts_real = _load_counts(lum, 'train', 'real', w_real_scaled=w_real)
    counts_shuf = _load_counts(lum, 'train', 'shuf', w_shuf_scaled=w_shuf)
    print('=== 仿真：真实脑 / 打乱脑（评测组） ===', flush=True)
    counts_real_ev = _load_counts(lum_ev, 'eval', 'real', w_real_scaled=w_real)
    counts_shuf_ev = _load_counts(lum_ev, 'eval', 'shuf', w_shuf_scaled=w_shuf)

    groups = make_groups(len(lum), args.seeds, N_PER_SEED)
    tr = frames_of_group(groups, TRAIN_SEEDS)
    va = frames_of_group(groups, VAL_SEEDS)
    # 防呆：划分必须无重叠（bug E）
    split_by_group(groups, TRAIN_SEEDS, EVAL_SEEDS, val_ids=VAL_SEEDS)

    Y_tr, Y_va, Y_te = acts[tr], acts[va], acts_ev
    eval_groups = make_groups(len(acts_ev), EVAL_SEEDS, N_PER_SEED)

    # 每臂独立构建特征（bug G：各自在训练组选 topK / 拟合 PCA / 拟合统计）
    idx_real = select_topk(counts_real[tr], Y_tr, K, ncls=NCLS)
    mu, sd = fit_stats(counts_real[tr][:, idx_real])
    idx_shuf = select_topk(counts_shuf[tr], Y_tr, K, ncls=NCLS)
    mu_s, sd_s = fit_stats(counts_shuf[tr][:, idx_shuf])
    Vt, mean = pca_fit(lum[tr], K)
    mu_p, sd_p = fit_stats(pca_apply(lum[tr], Vt, mean))

    F_tr = {'A_real_brain_topK': normalize(counts_real[tr][:, idx_real], mu, sd),
            'C_shuf_brain_topK': normalize(counts_shuf[tr][:, idx_shuf], mu_s, sd_s),
            'B_pca_pixelsK': normalize(pca_apply(lum[tr], Vt, mean), mu_p, sd_p)}
    F_va = {'A_real_brain_topK': normalize(counts_real[va][:, idx_real], mu, sd),
            'C_shuf_brain_topK': normalize(counts_shuf[va][:, idx_shuf], mu_s, sd_s),
            'B_pca_pixelsK': normalize(pca_apply(lum[va], Vt, mean), mu_p, sd_p)}
    F_te = {'A_real_brain_topK': normalize(counts_real_ev[:, idx_real], mu, sd),
            'C_shuf_brain_topK': normalize(counts_shuf_ev[:, idx_shuf], mu_s, sd_s),
            'B_pca_pixelsK': normalize(pca_apply(lum_ev, Vt, mean), mu_p, sd_p)}

    results = {'n_frames': N_FRAMES, 'teacher': 'corridor-v6', 'scenario': 'corridor_avoid',
               'train_seeds': TRAIN_SEEDS, 'val_seeds': VAL_SEEDS, 'eval_seeds': EVAL_SEEDS,
               'n_per_seed': N_PER_SEED, 'label_majority_frac': round(maj, 3), 'ncls': NCLS,
               'topk_note': '每臂在训练组独立选择: 0.5*rankpct(std)+0.5*rankpct(action_F6)',
               'arms': {}, 'gain_vs_pca': {}, 'gain_vs_shuf': {}, 'run_id': rid}

    arm_models = {}
    for name in F_tr:
        r = build_arm(name, None, F_tr[name], F_va[name], F_te[name],
                      Y_tr, Y_va, Y_te, eval_groups, EVAL_SEEDS)
        results['arms'][name] = r
        arm_models[name] = (train_linear(F_tr[name], Y_tr), None)

    # 配对增益（bug F：按 held-out seed 逐 seed 计算差值，再取均值±std）
    def paired_gain(name_a, name_b):
        Wa, ba = arm_models[name_a][0]
        Wb, bb = arm_models[name_b][0]
        diffs = []
        for s in EVAL_SEEDS:
            m = eval_groups == s
            pa = predict_linear(Wa, ba, F_te[name_a][m])
            pb = predict_linear(Wb, bb, F_te[name_b][m])
            ma = classification_metrics(Y_te[m], pa, NCLS)
            mb = classification_metrics(Y_te[m], pb, NCLS)
            diffs.append(ma['balanced_acc'] - mb['balanced_acc'])
        d = np.asarray(diffs)
        return {'linear_balanced_gain_mean': round(float(d.mean()), 3),
                'linear_balanced_gain_std': round(float(d.std()), 3),
                'per_seed_gains': [round(float(x), 3) for x in d]}

    results['gain_vs_pca'] = paired_gain('A_real_brain_topK', 'B_pca_pixelsK')
    results['gain_vs_shuf'] = paired_gain('A_real_brain_topK', 'C_shuf_brain_topK')
    print(f"gain A-B (real vs pca):  {results['gain_vs_pca']}", flush=True)
    print(f"gain A-C (real vs shuf): {results['gain_vs_shuf']}", flush=True)

    # bug L：结果 + provenance
    (run_dir / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                          encoding='utf-8')
    write_provenance(run_dir, config={'run_id': rid, 'script': 'corridor_ablation.py',
                                      'train_seeds': TRAIN_SEEDS, 'val_seeds': VAL_SEEDS,
                                      'eval_seeds': EVAL_SEEDS, 'n_per_seed': N_PER_SEED,
                                      'k': K, 'ncls': NCLS, 'shuf_seed': SHUF_SEED},
                     sources={'teacher': TEACHER, 'brain_graph': BRAIN,
                              'corridor_wad': WAD, 'script': Path(__file__)},
                     extra={'summary': {k: results['arms'][k]['linear']['balanced_acc'] for k in results['arms']}})
    (OUT / 'latest.json').write_text(json.dumps(
        {'run_id': rid, 'results': str(run_dir / 'results.json'),
         'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print('saved', run_dir / 'results.json', flush=True)


# ---------------------------------------------------------------------------
# 兼容导出（供 corridor_student.py 与旧探针脚本使用；新代码请用 corridor_common）
# ---------------------------------------------------------------------------
def sim_with_weights(lum_seq, weight_mod=None, name=''):
    """旧签名兼容：按注入的 weight 变换（None=真实权重）仿真，返回 (T,n) counts。"""
    brain = Brain(BRAIN)
    w = brain.weight.copy()
    if weight_mod is not None:
        w = weight_mod(w)
    w_scaled = (w * SCALE).astype(np.float32)
    return _sim(lum_seq, w_scaled, name=name)


def select_top32(counts, acts):
    """旧签名兼容：数据驱动 top32（6 类动作）。"""
    return select_topk(counts, acts, 32, ncls=NCLS)


def balanced(Y, pred):
    """旧签名兼容：6 类已出现类别的平均 recall。"""
    return classification_metrics(Y, pred, NCLS)['balanced_acc']


def zscore(F):
    """旧签名兼容：整段拟合统计再归一化。注意：新管线勿用（bug A/D）。"""
    mu, sd = fit_stats(F)
    return normalize(F, mu, sd)


if __name__ == '__main__':
    main()