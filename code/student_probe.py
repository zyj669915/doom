# -*- coding: utf-8 -*-
"""学生瓶颈定位探针（不跑闭环，只做离线评估）：

  A. 感光直读学生：lum PCA32 → 3 类转向       （不经脑：数据/任务可学性上限）
  B. 脑单帧学生：   counts top32 单帧 → 3 类     （现学生，对照 0.382）
  C. 脑5帧窗口学生：counts top32 最近5帧拼接 → 3 类（时间窗口是否有增益）

所有比较同数据集（2000 帧）、同评估（5 分割 seed 平衡准确率）。
"""
import sys
import numpy as np
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from corridor_student import OUT, balanced3, linear3_acc, train_mlp3, zscore_local  # noqa: E402
from corridor_ablation import pca_fit, pca_apply  # noqa: E402


def eval_cv(build_feats, name):
    lum = np.load(OUT / 'lum.npy')
    acts = np.load(OUT / 'acts.npy')
    acts3 = (acts // 2).astype(np.int64)
    F = build_feats(lum)
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
    print(f'{name}: linear {lin.mean():.3f}+-{lin.std():.3f}  mlp {mlp.mean():.3f}+-{mlp.std():.3f}', flush=True)
    return lin.mean(), mlp.mean()


# A. 感光 PCA32
lum = np.load(OUT / 'lum.npy')
acts = np.load(OUT / 'acts.npy')
acts3 = (acts // 2).astype(np.int64)
P, mu_pca = pca_fit(lum, 32)
F_pca = zscore_local(pca_apply(lum, P, mu_pca))
eval_cv(lambda l: zscore_local(pca_apply(l, P, mu_pca)), 'A. 感光 PCA32')

# B. 脑单帧（对照）
counts = np.load(OUT / 'counts-real.npy')
TOP32 = np.load(OUT / 'selected_sets.npz')['top32_raw']
F1 = zscore_local(counts[:, TOP32].astype(np.float32))
eval_cv(lambda l: F1, 'B. 脑 top32 单帧')

# C. 脑 5 帧窗口（最近 5 帧拼接，不足帧用首帧填充）
T, n = counts.shape
d32 = len(TOP32)
F5 = np.zeros((T, 5 * d32), np.float32)
for t in range(T):
    for j in range(5):
        idx = max(0, t - 4 + j)
        F5[t, j * d32:(j + 1) * d32] = counts[idx, TOP32]
F5 = zscore_local(F5)
eval_cv(lambda l: F5, 'C. 脑 top32 5帧窗口')
