# -*- coding: utf-8 -*-
"""读出升级探针：topK × 跨帧差分 × 时间窗口 的离线可学性对比（3 类转向）。

对照组：G1 top32 单帧 = 当前学生基线（0.382）。
候选：   G2 top128 单帧；G3 top128 5帧；G4 top128 5帧+差分；G5 top128 单帧+差分；
        G6 top256 5帧+差分；G7 top512 5帧+差分（观察收益饱和点）。

同数据（counts-real.npy 2000 帧）、同评估（5 分割 seed 平衡准确率）。
"""
import sys
import numpy as np
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from corridor_student import OUT, balanced3, linear3_acc, train_mlp3, zscore_local  # noqa: E402
from corridor_ablation import action_F, rankpct  # noqa: E402


def select_topk(counts, acts, k):
    ts = counts.std(0)
    Fa = action_F(counts, acts, ncls=3)
    score = 0.5 * rankpct(ts) + 0.5 * rankpct(Fa)
    return np.argsort(-score)[:k].astype(np.int64)


def build(counts, idx, win, diff):
    """win 帧窗口（不足用首帧填充），可选拼接跨帧差分。返回 (T, D)。"""
    T = counts.shape[0]
    d = len(idx)
    n_comp = 2 if diff else 1
    F = np.zeros((T, win * d * n_comp), np.float32)
    for t in range(T):
        col = 0
        for j in range(win):
            i0 = max(0, t - win + 1 + j)
            v = counts[i0, idx]
            F[t, col:col + d] = v
            col += d
            if diff:
                v0 = counts[i0 - 1, idx] if i0 > 0 else np.zeros(d, np.float32)
                F[t, col:col + d] = v - v0
                col += d
    return F


def eval_cv(F, acts3, name):
    lin, mlp = [], []
    for s in [0, 1, 2, 3, 4]:
        lin.append(linear3_acc(F, acts3, s))
        rng = np.random.RandomState(s)
        idx = rng.permutation(len(F)); ntr = int(0.8 * len(F))
        Xtr, Ytr = F[idx[:ntr]], acts3[idx[:ntr]]
        Xva, Yva = F[idx[ntr:]], acts3[idx[ntr:]]
        _, acc = train_mlp3(Xtr, Ytr, Xva, Yva, s, F.shape[1])
        mlp.append(acc)
    lin = np.array(lin); mlp = np.array(mlp)
    print(f'{name:34s} dim={F.shape[1]:5d}  linear {lin.mean():.3f}+-{lin.std():.3f}  '
          f'mlp {mlp.mean():.3f}+-{mlp.std():.3f}', flush=True)
    return lin.mean(), mlp.mean()


acts = np.load(OUT / 'acts.npy')
acts3 = (acts // 2).astype(np.int64)
counts = np.load(OUT / 'counts-real.npy')
maj = np.bincount(acts3, minlength=3).max() / len(acts3)
print(f'counts {counts.shape}  maj={maj:.3f}', flush=True)

res = {}


def _main():
    for k in (32, 128, 256, 512):
        idx = select_topk(counts, acts3, k)
        for win, diff, tag in ((1, False, f'G top{k} 单帧'),
                               (5, False, f'G top{k} 5帧'),
                               (5, True, f'G top{k} 5帧+差分'),
                               (1, True, f'G top{k} 单帧+差分')):
            if k == 32 and (win, diff) not in ((1, False),):
                continue
            F = build(counts, idx, win, diff)
            F = zscore_local(F)
            a, b = eval_cv(F, acts3, tag)
            res[tag] = {'linear': round(a, 3), 'mlp': round(b, 3)}
    print('RESULT', res, flush=True)


if __name__ == '__main__':
    _main()
