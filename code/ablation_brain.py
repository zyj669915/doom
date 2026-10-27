# -*- coding: utf-8 -*-
"""脑 vs 无脑 消融：真实连接组是否提供可重复增益？

三臂（同一批 500 帧、同一老师 distillable、同一学生结构、同一评测）：
  A 真实脑：counts(real)[:, top32]          —— 冻结连接组的神经活动
  B 直接特征：感光值 PCA→32 维                —— 无脑对照（人工前端直接喂学生）
  C 打乱脑：weight-shuffle 后 counts(shuf)[:, top32] —— 破坏结构对照（保留度分布）

学生：线性 softmax（纯 numpy）+ 2 层 MLP（torch），5 个分割 seed 求平衡准确率均值±std。
指标：12 类平衡准确率 vs 多数类基线；结论判定 A 是否显著优于 B/C。
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain
from doom.game import Game, retinal_samples
from readout_experiment import agc, motion, custom_step, readout_indexes, SCALE, BIAS, THRESH, DUR_MS, BRAIN, MANIFEST
from teacher_ppo import Policy, ObsBuilder, index_to_action_12

OUT = CODE / 'outputs' / 'doom' / 'ablation-brain'
OUT.mkdir(parents=True, exist_ok=True)
TEACHER = CODE / 'outputs' / 'doom' / 'teacher-v4-final' / 'teacher-distillable.pt'
SELECTED = np.load(CODE / 'outputs' / 'doom' / 'readout-exp' / 'selected_sets.npz')
TOP32 = SELECTED['top32_raw'].astype(np.int64)

SEEDS = [41027, 41028]
N_PER_SEED = 250
N_FRAMES = N_PER_SEED * len(SEEDS)
SHUF_SEED = 42
EVAL_SEEDS = [0, 1, 2, 3, 4]
MLP_EPOCHS = 40
MLP_LR = 1e-3
DEV = 'cpu'


# ---------- 数据采集（帧 + 感光 + 老师动作） ----------
UV = np.load(BRAIN)['uv']


def collect(teacher_path, seeds, n_per_seed):
    ck = torch.load(teacher_path, map_location='cpu', weights_only=True)
    cfg = ck.get('cfg', {})
    obs_mode = cfg.get('obs_mode', 'A')
    policy = Policy(obs_dim={'A': 24 * 32 * 2, 'B': 24 * 32 * 4, 'C': 24 * 32 * 3}[obs_mode], action_mode='12')
    policy.load_state_dict(ck['policy'])
    policy.eval()
    lum_l, act_l = [], []
    for seed in seeds:
        game = Game(seed=seed, scenario='combat_survival')
        game.new_episode()
        ob = ObsBuilder(mode=obs_mode, rand=0.0)
        for i in range(n_per_seed):
            if game.game.is_episode_finished():
                game.new_episode()
                ob = ObsBuilder(mode=obs_mode, rand=0.0)
            rgb = game.pixels()
            lum = retinal_samples(rgb, UV).astype(np.float32)
            lum_l.append(lum)
            o = ob.step(game) if i > 0 else ob.reset(game)
            with torch.no_grad():
                a = int(policy.act(torch.from_numpy(o).unsqueeze(0))[0].item())
            act_l.append(a)
            game.act(index_to_action_12(a))
        game.close()
        print(f'[collect seed={seed}] {n_per_seed} frames', flush=True)
    return np.stack(lum_l), np.asarray(act_l)


# ---------- 全脑仿真（可注入 weight 修改） ----------
def sim_with_weights(lum_seq, weight_mod=None, name=''):
    brain = Brain(BRAIN)
    w = brain.weight.copy()
    if weight_mod is not None:
        w = weight_mod(w)
    brain.weight = (w * SCALE).astype(np.float32)
    T = len(lum_seq)
    counts = np.zeros((T, brain.n), dtype=np.float32)
    prev = None
    t0 = time.perf_counter()
    for i, l in enumerate(lum_seq):
        l2 = agc(motion(l, prev))
        prev = l
        c = custom_step(brain, l2, DUR_MS, bias=BIAS, thresh=THRESH)
        counts[i] = c.astype(np.float32)
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f'  [{name}] {i + 1}/{T} frames, {el:.1f}s', flush=True)
    return counts


# ---------- 特征构造 ----------
def pca_fit(X, k):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k], X.mean(0)


def pca_apply(X, Vt, mean):
    return (X - mean) @ Vt.T


# ---------- 学生模型 ----------
class MLP(nn.Module):
    def __init__(self, d_in, ncls=12, h=64):
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
            acc = balanced(Yva, pred)
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def balanced(Y, pred):
    ncls = 12
    rec = []
    for c in range(ncls):
        m = Y == c
        if m.sum() == 0:
            continue
        rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


def linear_softmax_acc(F, acts, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(F)); ntr = int(0.8 * len(F))
    Xtr, Ytr = F[idx[:ntr]].astype(np.float64), acts[idx[:ntr]]
    Xte, Yte = F[idx[ntr:]].astype(np.float64), acts[idx[ntr:]]
    ncls = 12
    W = np.zeros((F.shape[1], ncls), np.float64); b = np.zeros(ncls, np.float64)
    for _ in range(800):
        logits = Xtr @ W + b
        ex = np.exp(logits - logits.max(1, keepdims=True))
        P = ex / ex.sum(1, keepdims=True)
        G = P.copy(); G[np.arange(ntr), Ytr] -= 1.0
        W -= 0.05 * (Xtr.T @ G / ntr + 1e-4 * W)
        b -= 0.05 * G.sum(0) / ntr
    pred = (Xte @ W + b).argmax(1)
    return balanced(Yte, pred)


def main():
    print('=== 脑 vs 无脑 消融：数据采集 ===', flush=True)
    lum, acts = collect(TEACHER, SEEDS, N_PER_SEED)
    np.save(OUT / 'lum.npy', lum)
    np.save(OUT / 'acts.npy', acts)
    print(f'lum {lum.shape} acts {acts.shape}, label dist {np.bincount(acts, minlength=12).tolist()}', flush=True)
    maj = float(np.bincount(acts, minlength=12).max() / len(acts))

    print('=== A 臂：真实脑仿真 ===', flush=True)
    t0 = time.perf_counter()
    counts_real = sim_with_weights(lum, weight_mod=None, name='real')
    np.save(OUT / 'counts-real.npy', counts_real)
    print(f'  real sim {time.perf_counter() - t0:.1f}s', flush=True)

    print('=== C 臂：打乱连接组（weight-shuffle, seed 42） ===', flush=True)
    t0 = time.perf_counter()
    rng = np.random.RandomState(SHUF_SEED)


    def shuffle_w(w):
        return rng.permutation(w)

    counts_shuf = sim_with_weights(lum, weight_mod=shuffle_w, name='shuf')
    np.save(OUT / 'counts-shuf.npy', counts_shuf)
    print(f'  shuf sim {time.perf_counter() - t0:.1f}s', flush=True)

    # 特征
    F_real = counts_real[:, TOP32].astype(np.float32)          # A
    F_shuf = counts_shuf[:, TOP32].astype(np.float32)          # C
    Vt, mean = pca_fit(lum, 32)                                 # B（在全量数据上拟合，分割只影响评测）
    F_pca = pca_apply(lum, Vt, mean).astype(np.float32)
    # 特征标准化（学生友好）
    def zscore(F):
        mu, sd = F.mean(0), F.std(0) + 1e-6
        return (F - mu) / sd
    F_real, F_shuf, F_pca = zscore(F_real), zscore(F_shuf), zscore(F_pca)

    arms = {'A_real_brain_top32': F_real, 'B_pca_pixels32': F_pca, 'C_shuf_brain_top32': F_shuf}
    results = {'n_frames': N_FRAMES, 'teacher': 'distillable', 'scenarios': 'combat_survival',
               'seeds': SEEDS, 'label_majority_frac': round(maj, 3), 'arms': {}}

    for aname, F in arms.items():
        lin, mlp = [], []
        for s in EVAL_SEEDS:
            lin.append(linear_softmax_acc(F, acts, s))
            rng = np.random.RandomState(s)
            idx = rng.permutation(len(F)); ntr = int(0.8 * len(F))
            Xtr, Ytr = F[idx[:ntr]], acts[idx[:ntr]]
            Xva, Yva = F[idx[ntr:]], acts[idx[ntr:]]
            _, acc = train_mlp(Xtr, Ytr, Xva, Yva, s, F.shape[1])
            mlp.append(acc)
        lin = np.array(lin); mlp = np.array(mlp)
        results['arms'][aname] = {
            'linear_balanced_mean': round(float(lin.mean()), 3),
            'linear_balanced_std': round(float(lin.std()), 3),
            'mlp_balanced_mean': round(float(mlp.mean()), 3),
            'mlp_balanced_std': round(float(mlp.std()), 3),
        }
        print(f'[{aname}] linear {lin.mean():.3f}±{lin.std():.3f}  mlp {mlp.mean():.3f}±{mlp.std():.3f}  (maj {maj:.3f})', flush=True)

    # 增益判定（同 seed 配对差）
    def paired_gain(Fa, Fb, kind):
        diffs = []
        for s in EVAL_SEEDS:
            if kind == 'linear':
                diffs.append(linear_softmax_acc(Fa, acts, s) - linear_softmax_acc(Fb, acts, s))
            else:
                rng = np.random.RandomState(s)
                idx = rng.permutation(len(Fa)); ntr = int(0.8 * len(Fa))
                Xtr, Ytr = Fa[idx[:ntr]], acts[idx[:ntr]]
                Xva, Yva = Fa[idx[ntr:]], acts[idx[ntr:]]
                _, a1 = train_mlp(Xtr, Ytr, Xva, Yva, s, Fa.shape[1])
                rng = np.random.RandomState(s)
                idx = rng.permutation(len(Fb)); ntr = int(0.8 * len(Fb))
                Xtr, Ytr = Fb[idx[:ntr]], acts[idx[:ntr]]
                Xva, Yva = Fb[idx[ntr:]], acts[idx[ntr:]]
                _, a2 = train_mlp(Xtr, Ytr, Xva, Yva, s, Fb.shape[1])
                diffs.append(a1 - a2)
        d = np.array(diffs)
        return round(float(d.mean()), 3), round(float(d.std()), 3)

    g1l = paired_gain(F_real, F_pca, 'linear'); g1m = paired_gain(F_real, F_pca, 'mlp')
    g2l = paired_gain(F_real, F_shuf, 'linear'); g2m = paired_gain(F_real, F_shuf, 'mlp')
    results['gain_vs_pca'] = {'linear': g1l, 'mlp': g1m}
    results['gain_vs_shuf'] = {'linear': g2l, 'mlp': g2m}
    print(f'gain A-B (real vs pca): linear {g1l}  mlp {g1m}', flush=True)
    print(f'gain A-C (real vs shuf): linear {g2l}  mlp {g2m}', flush=True)

    json.dump(results, open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
    print('saved', OUT / 'results.json')


if __name__ == '__main__':
    main()
