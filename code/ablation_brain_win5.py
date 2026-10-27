# -*- coding: utf-8 -*-
"""消融补充：5 帧窗口特征（t-4..t）三臂对比。
复用已存 counts-real/counts-shuf/lum/acts，仅构造窗口特征并重训学生，成本几分钟。
判定：若 A(真实脑) 在窗口下仍 ≈ B(直接特征) / C(打乱脑)，则连接组无增益结论成立。
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

OUT = CODE / 'outputs' / 'doom' / 'ablation-brain'
SELECTED = np.load(CODE / 'outputs' / 'doom' / 'readout-exp' / 'selected_sets.npz')
TOP32 = SELECTED['top32_raw'].astype(np.int64)
WIN = 5
EVAL_SEEDS = [0, 1, 2, 3, 4]
MLP_EPOCHS = 40
MLP_LR = 1e-3

from ablation_brain import balanced, linear_softmax_acc, train_mlp


def zscore(F):
    mu, sd = F.mean(0), F.std(0) + 1e-6
    return (F - mu) / sd


def window(F, w=WIN):
    T = len(F)
    out = np.stack([F[t - w + 1: t + 1] for t in range(w - 1, T)])
    return out


def main():
    lum = np.load(OUT / 'lum.npy')
    acts = np.load(OUT / 'acts.npy')
    counts_real = np.load(OUT / 'counts-real.npy')
    counts_shuf = np.load(OUT / 'counts-shuf.npy')

    F_real = counts_real[:, TOP32].astype(np.float32)
    F_shuf = counts_shuf[:, TOP32].astype(np.float32)

    # PCA 直接特征（在全量上拟合，与主实验一致）
    from ablation_brain import pca_fit, pca_apply
    Vt, mean = pca_fit(lum, 32)
    F_pca = pca_apply(lum, Vt, mean).astype(np.float32)

    arms = {'A_real_brain_top32': F_real, 'B_pca_pixels32': F_pca, 'C_shuf_brain_top32': F_shuf}
    results = {'win': WIN, 'note': '5-frame window, t-4..t', 'majority_frac': round(float(np.bincount(acts, minlength=12).max() / len(acts)), 3), 'arms': {}}

    for aname, F in arms.items():
        Fw = window(F)
        acts_w = acts[WIN - 1:]
        print(f'[{aname}] win-feat {Fw.shape}, labels {len(acts_w)}', flush=True)
        Fw = zscore(Fw.reshape(Fw.shape[0], -1))
        lin, mlp = [], []
        for s in EVAL_SEEDS:
            lin.append(linear_softmax_acc(Fw, acts_w, s))
            rng = np.random.RandomState(s)
            idx = rng.permutation(len(Fw)); ntr = int(0.8 * len(Fw))
            Xtr, Ytr = Fw[idx[:ntr]], acts_w[idx[:ntr]]
            Xva, Yva = Fw[idx[ntr:]], acts_w[idx[ntr:]]
            _, acc = train_mlp(Xtr, Ytr, Xva, Yva, s, Fw.shape[1])
            mlp.append(acc)
        lin, mlp = np.array(lin), np.array(mlp)
        results['arms'][aname] = {
            'linear_balanced_mean': round(float(lin.mean()), 3),
            'linear_balanced_std': round(float(lin.std()), 3),
            'mlp_balanced_mean': round(float(mlp.mean()), 3),
            'mlp_balanced_std': round(float(mlp.std()), 3),
        }
        print(f'  linear {lin.mean():.3f}±{lin.std():.3f}  mlp {mlp.mean():.3f}±{mlp.std():.3f}', flush=True)

    def paired(Fa, Fb, kind):
        d = []
        for s in EVAL_SEEDS:
            if kind == 'linear':
                d.append(linear_softmax_acc(Fa, acts_w, s) - linear_softmax_acc(Fb, acts_w, s))
            else:
                rng = np.random.RandomState(s)
                idx = rng.permutation(len(Fa)); ntr = int(0.8 * len(Fa))
                _, a1 = train_mlp(Fa[idx[:ntr]], acts_w[idx[:ntr]], Fa[idx[ntr:]], acts_w[idx[ntr:]], s, Fa.shape[1])
                rng = np.random.RandomState(s)
                idx = rng.permutation(len(Fb)); ntr = int(0.8 * len(Fb))
                _, a2 = train_mlp(Fb[idx[:ntr]], acts_w[idx[:ntr]], Fb[idx[ntr:]], acts_w[idx[ntr:]], s, Fb.shape[1])
                d.append(a1 - a2)
        d = np.array(d)
        return round(float(d.mean()), 3), round(float(d.std()), 3)

    results['gain_vs_pca'] = {'linear': None, 'mlp': None}
    # 重新构造窗口特征用于配对
    Fw_real = zscore(window(F_real).reshape(-1, WIN * 32))
    Fw_pca = zscore(window(F_pca).reshape(-1, WIN * 32))
    Fw_shuf = zscore(window(F_shuf).reshape(-1, WIN * 32))
    results['gain_vs_pca'] = {'linear': paired(Fw_real, Fw_pca, 'linear'), 'mlp': paired(Fw_real, Fw_pca, 'mlp')}
    results['gain_vs_shuf'] = {'linear': paired(Fw_real, Fw_shuf, 'linear'), 'mlp': paired(Fw_real, Fw_shuf, 'mlp')}
    print('gain A-B:', results['gain_vs_pca'], '  gain A-C:', results['gain_vs_shuf'], flush=True)

    json.dump(results, open(OUT / 'results-win5.json', 'w'), ensure_ascii=False, indent=1)
    print('saved', OUT / 'results-win5.json')


if __name__ == '__main__':
    main()
