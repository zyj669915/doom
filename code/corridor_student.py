# -*- coding: utf-8 -*-
"""走廊学生蒸馏 + 闭环评估（v2：3 类转向；bug A/C/D/E/F/H/K/L 修复）。

任务压缩为 3 类转向决策 {turn=-15/0/+15} + 前进恒定 20 —— 匹配果蝇飞行中
"始终前进、转弯是唯一视觉决策"的生态特性，也匹配老师行为（永不停止）。

修复记录（对照报告 §5.1 / §8.5）：
  * bug A：mu/sd 只在【训练组原始特征】上拟合，checkpoint 自足保存；
          闭环用同一组统计归一化原始 counts（不再「先 zscore 再存 0/1」）。
  * bug C/D：topK 选择、z-score 统计只在训练组（TRAIN_SEEDS）拟合；
  * bug E/F：按 seed 分组划分（训练/选型/评测三组不相交）；评测用独立
            held-out seeds（默认 41031/41032），不再同批数据重洗 5 次；
  * bug H：balanced accuracy 对 1/C、ordinary accuracy 对多数类，附混淆矩阵；
  * bug I：默认用纯像素老师（--teacher 指向 pixel 版本），ANGLE 老师仅作诊断；
  * bug K：闭环每个 episode 开头 reset_brain；
  * bug L：结果写 runs/<run_id>/ + provenance.json，不再原地覆盖。

用法：
  python corridor_teacher.py --use-angle 0 --total-steps 200000   # 先训纯像素老师
  python corridor_student.py --train 1 --n-eps 6 --collect-eval 1
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
from corridor_teacher import head_to_action  # noqa: E402
from corridor_ablation import collect, sim_with_weights  # noqa: E402  （兼容包装）
from corridor_common import (  # noqa: E402
    classification_metrics, fit_stats, frames_of_group, make_groups, normalize,
    reset_brain, run_id, select_topk, split_by_group, write_provenance,
)

OUT = CODE / 'outputs' / 'doom' / 'corridor-student'
OUT.mkdir(parents=True, exist_ok=True)
WAD = CODE.parent.parent / 'doomfly' / 'doom' / 'scenarios' / 'corridor_avoid.wad'
# bug I：主训练用纯像素老师；ANGLE 老师（旧 teacher-corridor-final.pt）仅作诊断上限
TEACHER_PIXEL = CODE / 'outputs' / 'doom' / 'corridor-teacher-v6' / 'teacher-corridor-pixel-final.pt'

COLLECT_SEEDS = [41027, 41028, 41029, 41030]   # 训练+选型
TRAIN_SEEDS = [41027, 41028, 41029]            # topK/统计/模型拟合
VAL_SEEDS = [41030]                            # 模型选择
EVAL_SEEDS = [41031, 41032]                    # 独立 held-out 评测（bug F）
N_PER_SEED = 500
NCLS = 3                    # 转向 3 类；fwd 恒定 20
MAX_CLOSED_TICS = 700
UV = np.load(BRAIN)['uv']


# ---------- 3 类学生工具 ----------
class MLP3(nn.Module):
    def __init__(self, d_in, ncls=NCLS, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, ncls))

    def forward(self, x):
        return self.net(x)


def train_linear3(Xtr, Ytr, iters=800, lr=0.05, reg=1e-4):
    """纯 numpy 线性 softmax，返回 (W, b)。"""
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


def train_mlp3(Xtr, Ytr, Xva, Yva, seed, d_in):
    torch.manual_seed(seed)
    model = MLP3(d_in).to('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, best_state = -1, None
    n = len(Xtr)
    for ep in range(40):
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


def train_mlp3_full(Xtr, Ytr, d_in, seed=42, epochs=60):
    torch.manual_seed(seed)
    model = MLP3(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.long)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, Yt[idx])
            loss.backward()
            opt.step()
    return model


def predict_mlp3(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


# ---------- 数据/仿真缓存（bug L：记录老师来源，老师换了必须重采） ----------
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
        print(f'=== 老师文件已变化，{tag} 数据需重新采集 ===', flush=True)
    if not Path(teacher).exists():
        raise SystemExit(
            f'纯像素老师不存在：{teacher}\n'
            f'请先训练：python corridor_teacher.py --use-angle 0 --total-steps 200000\n'
            f'（ANGLE 老师仅作诊断上限，学生主训练必须用纯像素老师，见报告 §5.1 I）')
    lum, acts = collect(teacher, seeds, N_PER_SEED, tag=tag)
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
    counts = sim_with_weights(lum, weight_mod=None, name=f'real-{tag}')
    np.save(p, counts)
    print(f'  sim {time.perf_counter() - t0:.1f}s', flush=True)
    return counts


def train_student(reuse=True, collect_eval=True, teacher=None):
    teacher = teacher or TEACHER_PIXEL
    rid = run_id('stu')
    run_dir = OUT / 'runs' / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    lum, acts6 = load_data('train', COLLECT_SEEDS, teacher)
    acts3 = (acts6 // 2).astype(np.int64)          # 6 类 → 3 类 turn
    maj = float(np.bincount(acts3, minlength=NCLS).max() / len(acts3))
    print(f'lum {lum.shape} turn3 dist {np.bincount(acts3, minlength=NCLS).tolist()} maj={maj:.3f}', flush=True)

    counts = load_counts(lum, 'train')
    groups = make_groups(len(counts), COLLECT_SEEDS, N_PER_SEED)
    tr = frames_of_group(groups, TRAIN_SEEDS)
    va = frames_of_group(groups, VAL_SEEDS)
    split_by_group(groups, TRAIN_SEEDS, EVAL_SEEDS, val_ids=VAL_SEEDS)  # 防呆（bug E）

    # bug C：topK 只在训练组选；bug D：统计只在训练组原始特征上拟合
    TOP32 = select_topk(counts[tr], acts3[tr], 32, ncls=NCLS)
    mu, sd = fit_stats(counts[tr][:, TOP32])
    F_tr = normalize(counts[tr][:, TOP32], mu, sd)
    F_va = normalize(counts[va][:, TOP32], mu, sd)
    Y_tr, Y_va = acts3[tr], acts3[va]

    # bug E/F：模型选择用选型组（va），不用随机 80/20 重洗
    W, b = train_linear3(F_tr, Y_tr)
    lin_va = classification_metrics(Y_va, predict_linear3(W, b, F_va), NCLS)
    _, mlp_va = train_mlp3(F_tr, Y_tr, F_va, Y_va, seed=0, d_in=F_tr.shape[1])
    use_mlp = mlp_va >= lin_va['balanced_acc'] - 0.01
    print(f'[select] linear va bal={lin_va["balanced_acc"]} mlp va bal={mlp_va:.4f} '
          f'-> chosen={ "mlp" if use_mlp else "linear" }', flush=True)

    # 最终模型：训练组+选型组合并训练
    if use_mlp:
        F_all = np.concatenate([F_tr, F_va], axis=0)
        Y_all = np.concatenate([Y_tr, Y_va], axis=0)
        model = train_mlp3_full(F_all, Y_all, F_all.shape[1])
        state = model.state_dict()
        kind = 'mlp'
    else:
        F_all = np.concatenate([F_tr, F_va], axis=0)
        Y_all = np.concatenate([Y_tr, Y_va], axis=0)
        W, b = train_linear3(F_all, Y_all, iters=1200)
        state = {'W': W.astype(np.float32), 'b': b.astype(np.float32)}
        kind = 'linear'

    ck_path = run_dir / 'student-final.pt'
    torch.save({'kind': kind, 'state': state, 'top32': TOP32, 'mu': mu, 'sd': sd,
                'win': 1, 'diff': False, 'run_id': rid,
                'train_seeds': TRAIN_SEEDS, 'val_seeds': VAL_SEEDS,
                'teacher': str(teacher), 'teacher_sha256': _teacher_sha(teacher)},
               ck_path)
    print('saved', ck_path, flush=True)

    # bug L：provenance + 指标落盘
    config = {'run_id': rid, 'script': 'corridor_student.py', 'task': 'turn3 + fwd const 20',
              'train_seeds': TRAIN_SEEDS, 'val_seeds': VAL_SEEDS, 'eval_seeds': EVAL_SEEDS,
              'n_per_seed': N_PER_SEED, 'ncls': NCLS, 'topk': int(len(TOP32))}
    write_provenance(run_dir, config,
                     sources={'teacher': teacher, 'brain_graph': BRAIN,
                              'corridor_wad': WAD, 'script': Path(__file__)})
    res = {'run_id': rid, 'kind': kind, 'teacher': str(teacher),
           'n_frames': len(counts), 'maj_frac': round(maj, 3),
           'linear_val': lin_va, 'mlp_val_balanced': round(float(mlp_va), 4),
           'chosen': kind, 'n_train': int(len(F_tr)), 'n_val': int(len(F_va))}
    (run_dir / 'train-results.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('train-results:', {k: v for k, v in res.items() if k != 'linear_val'}, flush=True)

    # bug F：held-out 评测（独立 seeds）
    if collect_eval:
        lum_ev, acts6_ev = load_data('eval', EVAL_SEEDS, teacher)
        acts3_ev = (acts6_ev // 2).astype(np.int64)
        counts_ev = load_counts(lum_ev, 'eval')
        F_ev = normalize(counts_ev[:, TOP32], mu, sd)
        if kind == 'mlp':
            pred = predict_mlp3(model, F_ev)
        else:
            W, b = state['W'], state['b']
            pred = predict_linear3(W, b, F_ev)
        ev_metrics = classification_metrics(acts3_ev, pred, NCLS)
        ev_res = {'run_id': rid, 'eval_seeds': EVAL_SEEDS,
                  'metrics': ev_metrics}
        (run_dir / 'eval-results.json').write_text(
            json.dumps(ev_res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'[held-out eval] balanced={ev_metrics["balanced_acc"]} '
              f'(chance {ev_metrics["balanced_chance"]}) '
              f'ordinary={ev_metrics["ordinary_acc"]} (maj {ev_metrics["majority_frac"]})', flush=True)

    (OUT / 'latest.json').write_text(json.dumps(
        {'run_id': rid, 'student': str(ck_path),
         'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    return kind, rid


# ---------- 闭环（bug K：每 episode 重置脑；bug A：用训练统计归一化原始特征） ----------
def closed_loop(n_eps, max_tics=MAX_CLOSED_TICS, seed=41031, ck_path=None):
    ck_path = ck_path or (OUT / 'runs' / (json.loads((OUT / 'latest.json').read_text())['run_id'])
                          / 'student-final.pt')
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    kind = ck['kind']
    TOP32, mu, sd = ck['top32'], ck['mu'], ck['sd']
    if kind == 'mlp':
        student = MLP3(len(TOP32))
        student.load_state_dict(ck['state'])
        student.eval()
    else:
        W, b = ck['state']['W'], ck['state']['b']

    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)

    rid = run_id('stu-cl')
    run_dir = OUT / 'runs' / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    game = Game(seed=seed, scenario='corridor_avoid')
    n_exit, n_wall_tics, total_tics = 0, 0, 0
    ep_info = []
    for ep in range(n_eps):
        game.new_episode()
        reset_brain(brain)          # bug K：每局独立
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
            F = normalize(c[TOP32], mu, sd)          # bug A：原始特征 + 训练统计
            with torch.no_grad():
                if kind == 'mlp':
                    logits = student(torch.from_numpy(F).float().unsqueeze(0)).squeeze(0)
                    ti = int(logits.argmax().item())
                else:
                    lgl = F.astype(np.float64) @ W + b
                    ti = int(np.argmax(lgl))
            x0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
            y0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
            game.act(head_to_action(ti, 1))       # fwd 恒定 20
            x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
            y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
            moved = math.hypot(x1 - x0, y1 - y0)
            if moved < 0.5:
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
        n_wall_tics += hits
        ep_info.append({'ep': ep, 'exit': exited, 'tics': tics, 'hits': hits,
                        'max_prog': round(max_prog, 3)})
        print(f'  ep={ep} exit={exited} tics={tics} hits={hits} maxprog={max_prog:.3f}', flush=True)
    game.close()
    res = {'student_kind': kind, 'n_eps': n_eps, 'max_tics': max_tics,
           'eval_seed': seed,
           'exit_rate': round(n_exit / n_eps, 3), 'exit_count': n_exit,
           'tics_mean': round(total_tics / n_eps, 1),
           'wall_rate': round(n_wall_tics / total_tics, 4) if total_tics else 0.0,
           'episodes': ep_info, 'run_id': rid,
           'student_ckpt': str(ck_path),
           'reference': {'teacher_exit_rate': 1.0, 'teacher_tics_mean': 565.2,
                         'teacher_wall_rate': 0.0007}}
    (run_dir / 'closed-loop-results.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    write_provenance(run_dir, {'run_id': rid, 'script': 'corridor_student.py::closed_loop',
                               'eval_seed': seed, 'n_eps': n_eps, 'max_tics': max_tics},
                     sources={'student_ckpt': ck_path, 'brain_graph': BRAIN,
                              'corridor_wad': WAD, 'script': Path(__file__)})
    (OUT / 'latest.json').write_text(json.dumps(
        {'run_id': rid, 'closed_loop': str(run_dir / 'closed-loop-results.json'),
         'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print('closed-loop:', {k: v for k, v in res.items() if k != 'episodes'}, flush=True)
    return res


# ---------- 兼容旧探针脚本的导出 ----------
def balanced3(Y, pred):
    """旧版：3 类已出现类别的平均 recall（与 classification_metrics 一致）。"""
    return classification_metrics(Y, pred, NCLS)['balanced_acc']


def zscore_local(F):
    """旧版工具（仅兼容 corridor_readout_upgrade 等探针）。注意：新管线勿用，
    统计必须在训练组原始特征上拟合（bug A/D）。"""
    mu, sd = fit_stats(F)
    return normalize(F, mu, sd)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=int, default=1, help='1=训练学生（复用已有数据/仿真）')
    p.add_argument('--n-eps', type=int, default=6)
    p.add_argument('--max-tics', type=int, default=MAX_CLOSED_TICS)
    p.add_argument('--teacher', type=str, default=str(TEACHER_PIXEL),
                   help='老师 checkpoint（默认纯像素版；ANGLE 版仅诊断）')
    p.add_argument('--collect-eval', type=int, default=1,
                   help='采集并仿真 held-out 评测组（bug F）')
    p.add_argument('--eval-seed', type=int, default=41031, help='闭环评测 seed（不在训练组）')
    args = p.parse_args()
    if args.train:
        _, rid = train_student(reuse=True, collect_eval=bool(args.collect_eval),
                               teacher=Path(args.teacher))
        print(f'chosen student run: {rid}', flush=True)
    closed_loop(args.n_eps, args.max_tics, seed=args.eval_seed)
