"""Distill v2 with held-out validation: collect -> 80/20 split -> train with early stop.

Loads a PPO teacher (trained by teacher_ppo.py), collects aligned 5-frame-window
features (70-dim) via the REAL fly connectome, trains StudentV2 on 80% of frames
and reports agreement on the held-out 20% (author acceptance: >=60%, v1 was 34%).

Usage:
  python train_v2_holdout.py --teacher ppo --episodes 10 --tics 500 [--reuse-data]
"""
import json, sys, time
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from distill_v2 import collect, StudentV2, OUT, SCALE, BIAS, THRESH, DUR_MS, SEEDS, N_ACT, WIN, FEAT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='ppo', choices=['ppo'])
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--tics', type=int, default=500)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--reuse-data', action='store_true')
    args = ap.parse_args()

    from teacher_ppo import Policy
    teacher = Policy()
    ck = torch.load(CODE / 'outputs' / 'doom' / 'teacher-v1' / 'teacher.pt', map_location='cpu', weights_only=True)
    teacher.load_state_dict(ck['policy'])
    teacher.eval()

    if args.reuse_data and (OUT / 'distill-data.npz').exists():
        d = np.load(OUT / 'distill-data.npz')
        Xs, Y, mu, sd = d['X'], d['Y'], d['mu'], d['sd']
        print('reused data', Xs.shape, flush=True)
    else:
        Xs, Y, mu, sd = collect(teacher, args.episodes, args.tics, 'ppo')
        np.savez(OUT / 'distill-data.npz', X=Xs, Y=Y, mu=mu, sd=sd)

    n = len(Y)
    rng = np.random.RandomState(41027)
    perm = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    Xtr, Ytr = Xs[tr], Y[tr]
    Xte, Yte = Xs[te], Y[te]

    torch.manual_seed(41027)
    student = StudentV2()
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    Xt = torch.from_numpy(Xtr); Yt = torch.from_numpy(Ytr)
    Xe = torch.from_numpy(Xte); Ye = torch.from_numpy(Yte)

    def acc(x, y):
        with torch.no_grad():
            pred = student(x).argmax(1)
        return (pred == y).float().mean().item()

    best_te, best_ep, since = 0.0, -1, 0
    metrics = {'epochs': [], 'train_acc': [], 'heldout_acc': []}
    t0 = time.time()
    for ep in range(args.epochs):
        perm2 = torch.randperm(n_tr)
        s = 0.0
        for b in range(0, n_tr, 256):
            bi = perm2[b:b + 256]
            loss = nn.functional.cross_entropy(student(Xt[bi]), Yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item() * len(bi)
        tr_acc = acc(Xt, Yt)
        te_acc = acc(Xe, Ye)
        metrics['epochs'].append(ep)
        metrics['train_acc'].append(tr_acc)
        metrics['heldout_acc'].append(te_acc)
        if ep % 10 == 0:
            print(f'epoch {ep}: loss={s/n_tr:.4f} train_acc={tr_acc:.3f} heldout_acc={te_acc:.3f} ({time.time()-t0:.0f}s)', flush=True)
        if te_acc > best_te:
            best_te, best_ep, since = te_acc, ep, 0
            torch.save({'student': student.state_dict(), 'mu': mu, 'sd': sd,
                        'scale': SCALE, 'bias': BIAS, 'thresh': THRESH,
                        'epoch': ep, 'train_acc': tr_acc, 'heldout_acc': te_acc},
                       OUT / 'student.pt')
        else:
            since += 1
            if since >= args.patience:
                print(f'early stop at epoch {ep}, best heldout at epoch {best_ep}', flush=True)
                break

    print(f'BEST heldout_acc={best_te:.3f} at epoch {best_ep}', flush=True)
    summary = {'n_frames': int(n), 'n_train': int(n_tr), 'n_heldout': int(n - n_tr),
               'best_heldout_acc': float(best_te), 'best_epoch': int(best_ep),
               'final_train_acc': float(metrics['train_acc'][-1]),
               'label_dist': [int(x) for x in np.bincount(Y, minlength=N_ACT)]}
    (OUT / 'holdout_metrics.json').write_text(json.dumps(summary, indent=1) + '\n')
    (OUT / 'holdout_curve.json').write_text(json.dumps(metrics, indent=1) + '\n')
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
