# -*- coding: utf-8 -*-
"""非蒸馏 Q-learning 短时训练：冻结真实全脑连接组，只训练读出层（Ponghiran 2019 式）。

与旧蒸馏管线的本质区别：
  * 训练信号 = 环境奖励（progress shaping / 撞墙惩罚 / 出口奖励），不是老师动作标签；
  * 探索数据 = 无老师偏置随机策略在线采集，全程全脑仿真；
  * 特征选择 = 无监督（神经元时间方差），不依赖动作标签；
  * 学习 = 离线 FQI（fitted Q iteration），只用连接组响应特征驱动 Doom。

效率设计（全脑仿真 0.54s/帧 是硬约束）：
  1) 探索采集：multiprocessing 并行 4 seeds × 500 帧 ≈ 5 分钟墙钟；
  2) 离线 FQI：纯 torch，<1 分钟；
  3) 闭环验证：3 episode × 400 tics ≈ 11 分钟。

用法：
  python corridor_rl_q.py                    # 全流程
  python corridor_rl_q.py --eval-only 1      # 只用已有模型闭环评估
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

# ---- 大脑：使用已核验的干净副本（与运行中的其他进程隔离）----
BRAIN_CLEAN = Path(r'D:\fly\experiment\code\outputs\doom\brain-v0-clean\malecns_v1\graph.npz')
OUT = CODE / 'outputs' / 'doom' / 'corridor-rl-q'
OUT.mkdir(parents=True, exist_ok=True)

# 生理参数（与 readout_experiment 校准一致，非默认值）
SCALE, BIAS, THRESH = 1.5, 6.0, -48.0
DUR_MS = 50.0

from numba import njit  # noqa: E402
from doom.game import Game, retinal_samples  # noqa: E402
from corridor_teacher import head_to_action, dist, at_exit  # noqa: E402
from corridor_common import (  # noqa: E402
    FeatureWindow, fit_stats, make_groups, normalize, reset_brain, run_id,
    select_topk_unsupervised, split_by_group, write_provenance,
)

MAX_TICS_COLLECT = 2000
WALL_PENALTY, PROGRESS_GAIN, EXIT_REWARD = -0.15, 0.01, 10.0
WIN_FQI, DIFF_FQI = 5, True


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


class Brain:
    """精简版：只加载干净副本，其余与 doom.engine.Brain 一致。"""

    def __init__(self, path):
        a = np.load(path)
        for k in ['ptr', 'post', 'weight', 'ids', 'retina', 'uv', 'lamina', 'sugar', 'superclass']:
            setattr(self, k, a[k])
        self.n = len(self.ids)
        self.dt = 0.1
        self.cursor = 0
        self.v = np.full(self.n, -52, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.drive = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.queue = np.zeros((int(round(1.8 / self.dt)) + 1, self.n), dtype=np.int32)
        self.queue_count = np.zeros(self.queue.shape[0], dtype=np.int32)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.luminance = np.zeros(len(self.retina), dtype=np.float32)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        initial = np.unique(np.r_[self.retina, self.lamina, self.sugar])
        self.active[:len(initial)] = initial
        self.active_flag[initial] = 1
        self.nactive = np.asarray([len(initial)], dtype=np.int32)
        self.sim_ms = 0.0

    def step(self, lum, duration_ms, bias=BIAS, thresh=THRESH):
        steps = int(round(duration_ms / self.dt))
        alpha = 1 - np.exp(-steps * self.dt / 10)
        self.luminance += alpha * (np.clip(lum, 0, 1) - self.luminance)
        self.drive.fill(0)
        self.drive[self.lamina] = bias
        self.drive[self.retina] = 30 * self.luminance / (0.02 + self.luminance)
        self.counts.fill(0)
        self.cursor = advance_th(self.ptr, self.post, self.weight, self.v, self.g,
            self.refractory, self.drive, self.queue, self.queue_count, self.cursor,
            steps, self.dt, self.counts, self.active, self.active_flag,
            self.nactive, thresh)
        self.sim_ms += steps * self.dt
        return self.counts.copy()


def agc(lum, eps=1e-4):
    sd = lum.std()
    if sd < eps:
        return lum.copy()
    return 0.5 + 0.5 * np.tanh((lum - lum.mean()) / (sd + eps) / 1.5)


def motion(lum, prev, k=1.2):
    if prev is None:
        return lum.copy()
    return np.clip(lum + k * (lum - prev), 0, 1)


def collect_one(seed, n_frames, uv, out_dir):
    """无老师偏置随机探索：60% 直行 / 20% 左 / 20% 右，fwd 恒 20。
    直接落盘 counts (T,n) float32、acts、rwds、dones、groups 到 out_dir。
    bug K：每个游戏 episode 开头 reset_brain；
    bug J：groups 记录 episode id，供 FQI 边界处理与窗口分组。
    """
    out_dir = Path(out_dir)
    brain = Brain(BRAIN_CLEAN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    rng = np.random.RandomState(seed)
    game = Game(seed=seed, scenario='corridor_avoid')
    game.new_episode()

    counts = np.zeros((n_frames, brain.n), dtype=np.float32)
    acts = np.zeros(n_frames, dtype=np.int64)
    rwds = np.zeros(n_frames, dtype=np.float32)
    dones = np.zeros(n_frames, dtype=bool)
    groups = np.zeros(n_frames, dtype=np.int64)
    prev = None
    ep_id = 0
    t0 = time.perf_counter()
    for i in range(n_frames):
        if game.game.is_episode_finished():
            game.new_episode()
            reset_brain(brain)          # bug K：每局独立
            prev = None
            ep_id += 1
        groups[i] = ep_id
        rgb = game.pixels()
        lum = retinal_samples(rgb, uv).astype(np.float32)
        l2 = agc(motion(lum, prev))
        prev = lum
        c = brain.step(l2, DUR_MS)
        counts[i] = c.astype(np.float32)

        r = rng.rand()
        turn_i = 1 if r < 0.6 else (0 if r < 0.8 else 2)
        acts[i] = turn_i
        x0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
        y0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
        game.act(head_to_action(turn_i, 1))
        x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
        y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
        moved = math.hypot(x1 - x0, y1 - y0)
        rw = PROGRESS_GAIN * (dist(x0, y0) - dist(x1, y1))
        if moved < 0.5:
            rw += WALL_PENALTY
        if at_exit(x1, y1):
            rw += EXIT_REWARD
        rwds[i] = rw
        dones[i] = at_exit(x1, y1) or game.game.is_episode_finished()
        if (i + 1) % 100 == 0:
            print(f'  [collect seed={seed}] {i + 1}/{n_frames} '
                  f'{time.perf_counter() - t0:.1f}s', flush=True)
    game.close()
    np.savez(out_dir / f'collect-{seed}.npz', counts=counts, acts=acts,
             rwds=rwds, dones=dones, groups=groups)
    del counts
    return None


def build_features_grouped(counts_all, groups, idx, win, diff):
    """win 帧窗口 + 差分，返回 (T, win*d*comp)。

    bug J 修复：按组（episode/seed）分别构建再拼接——窗口绝不跨组，
    与旧版「整段拼接后统一 build」相比，边界帧不再混入上一组内容。
    """
    w = FeatureWindow(idx, win=win, diff=diff)
    parts = []
    for g in np.unique(groups):
        m = groups == g
        parts.append(w.build(counts_all[m]))
    return np.concatenate(parts, axis=0)


def build_features(counts_all, idx, win, diff):
    """旧签名兼容（无组信息时按单组处理）。新代码请用 build_features_grouped。"""
    groups = np.zeros(len(counts_all), dtype=np.int64)
    return build_features_grouped(counts_all, groups, idx, win, diff)


class QNet(nn.Module):
    def __init__(self, d_in, n_act=3, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, n_act))

    def forward(self, x):
        return self.net(x)


def train_fqi(F, acts, rwds, dones, groups, rounds=25, lr=1e-3, gamma=0.99,
              batch=256, seed=7, device='cpu'):
    """离线 fitted Q iteration：只用 (s,a,r,s') 转移，目标 = 环境奖励。

    bug J 修复：groups 标识 episode/seed；每组最后一帧视为 terminal，
    s' 绝不跨组（不再出现「seed41027 末帧 → seed41028 首帧」的虚假转移）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(F)
    # s_t = F[t]，s_{t+1} = F[t+1]；组边界处无后继（D=1 使目标不含 Xn）
    is_last = np.zeros(n, dtype=bool)
    if n > 1:
        is_last[:-1] = groups[:-1] != groups[1:]
    is_last[-1] = True
    D = (dones | is_last).astype(np.float32)
    Xn = np.zeros_like(F)
    if n > 1:
        Xn[:-1] = F[1:]
    X = F
    A = acts.astype(np.int64)
    R = rwds

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Xnt = torch.tensor(Xn, dtype=torch.float32, device=device)
    At = torch.tensor(A, dtype=torch.long, device=device)
    Rt = torch.tensor(R, dtype=torch.float32, device=device)
    Dt = torch.tensor(D, dtype=torch.float32, device=device)

    q = QNet(F.shape[1]).to(device)
    qt = QNet(F.shape[1]).to(device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()

    log = []
    for rnd in range(rounds):
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            with torch.no_grad():
                qn = qt(Xnt[b]).max(1).values
                y = Rt[b] + gamma * qn * (1.0 - Dt[b])
            opt.zero_grad()
            loss = lossf(q(Xt[b]).gather(1, At[b].unsqueeze(1)).squeeze(1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        if (rnd + 1) % 5 == 0:
            qt.load_state_dict(q.state_dict())
        log.append({'round': rnd + 1, 'loss': round(tot / cnt, 4)})
        print(f'  [fqi round={rnd + 1}] loss={tot / cnt:.4f}', flush=True)
    return q, log


def closed_loop(model, uv, n_eps=3, max_tics=400, seed=41031, kind='q',
                idx=None, mu=None, sd=None, device='cpu'):
    """greedy 闭环：Q argmax 驱动游戏，fwd 恒 20。

    bug B：窗口用 FeatureWindow 在线构建（与离线逐元素一致）；
    bug K：每个 episode 开头 reset_brain；
    bug L：结果写 runs/<run_id>/ + provenance.json。
    """
    d = len(idx)
    win, diff = WIN_FQI, DIFF_FQI
    w = FeatureWindow(idx, win=win, diff=diff)
    brain = Brain(BRAIN_CLEAN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    game = Game(seed=seed, scenario='corridor_avoid')
    n_exit, n_wall, total_tics = 0, 0, 0
    ep_info = []
    t0 = time.perf_counter()
    rid = run_id('rlq-cl')
    run_dir = OUT / 'runs' / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(n_eps):
        game.new_episode()
        reset_brain(brain)      # bug K
        w.reset()               # bug B：窗口按 episode 独立
        prev = None
        tics, hits, max_prog, exited = 0, 0, 0.0, False
        while tics < max_tics:
            if game.game.is_episode_finished():
                break
            rgb = game.pixels()
            lum = retinal_samples(rgb, uv).astype(np.float32)
            l2 = agc(motion(lum, prev))
            prev = lum
            c = brain.step(l2, DUR_MS)
            feat = w.step(c[idx].astype(np.float32))
            F = normalize(feat, mu, sd)          # 训练统计归一化原始特征
            with torch.no_grad():
                x = torch.from_numpy(F).float().unsqueeze(0).to(device)
                ti = int(model(x).argmax(1).item())
            x0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
            y0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
            game.act(head_to_action(ti, 1))
            x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
            y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
            if math.hypot(x1 - x0, y1 - y0) < 0.5:
                hits += 1
            if at_exit(x1, y1):
                exited = True
                break
            prog = 1.0 - dist(x1, y1) / 3456.0
            max_prog = max(max_prog, prog)
            tics += 1
        n_exit += int(exited)
        total_tics += tics
        n_wall += hits
        ep_info.append({'ep': ep, 'exit': exited, 'tics': tics, 'hits': hits,
                        'max_prog': round(max_prog, 3)})
        print(f'  ep={ep} exit={exited} tics={tics} hits={hits} maxprog={max_prog:.3f}', flush=True)
    game.close()
    res = {'n_eps': n_eps, 'eval_seed': seed,
           'exit_rate': round(n_exit / n_eps, 3), 'exit_count': n_exit,
           'tics_mean': round(total_tics / n_eps, 1),
           'wall_rate': round(n_wall / total_tics, 4) if total_tics else 0.0,
           'episodes': ep_info, 'run_id': rid,
           'elapsed_s': round(time.perf_counter() - t0, 1)}
    (run_dir / 'closed-loop-results.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    write_provenance(run_dir, {'run_id': rid, 'script': 'corridor_rl_q.py::closed_loop',
                               'eval_seed': seed, 'n_eps': n_eps, 'max_tics': max_tics,
                               'win': win, 'diff': diff},
                     sources={'brain_graph': BRAIN_CLEAN,
                              'corridor_wad': Path(r'D:\fly\doomfly\doom\scenarios\corridor_avoid.wad'),
                              'script': Path(__file__)})
    (OUT / 'latest.json').write_text(json.dumps(
        {'run_id': rid, 'closed_loop': str(run_dir / 'closed-loop-results.json'),
         'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print('closed-loop:', {k: v for k, v in res.items() if k != 'episodes'}, flush=True)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[41027, 41028, 41029, 41030])
    p.add_argument('--frames-per-seed', type=int, default=500)
    p.add_argument('--topk', type=int, default=512)
    p.add_argument('--rounds', type=int, default=25)
    p.add_argument('--eval-eps', type=int, default=3)
    p.add_argument('--eval-max-tics', type=int, default=400)
    p.add_argument('--eval-seed', type=int, default=41031,
                   help='闭环评测 seed（不在收集组，避免评测泄漏）')
    p.add_argument('--eval-only', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cpu', help='cpu / cuda')
    p.add_argument('--reuse-data', type=int, default=1, help='复用已落盘的探索数据')
    args = p.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARN: cuda requested but unavailable, fallback to cpu', flush=True)
        device = 'cpu'

    uv = np.load(BRAIN_CLEAN)['uv']

    if args.eval_only:
        latest = json.loads((OUT / 'latest.json').read_text(encoding='utf-8'))
        ck = torch.load(OUT / 'runs' / latest['run_id'] / 'model.pt',
                        map_location='cpu', weights_only=False)
        mu, sd, idx = ck['mu'], ck['sd'], ck['topk']
        model = QNet(ck['dim']).to(device)
        model.load_state_dict(ck['state'])
        model.eval()
        res = closed_loop(model, uv, args.eval_eps, args.eval_max_tics,
                          seed=args.eval_seed, idx=idx, mu=mu, sd=sd, device=device)
        res.update({'reference': {'teacher_exit_rate': 1.0, 'teacher_tics_mean': 565.2,
                                  'teacher_wall_rate': 0.0007},
                    'distill_student_exit_rate': 0.0})
        print('closed-loop:', {k: v for k, v in res.items() if k != 'episodes'}, flush=True)
        sys.exit(0)

    rid = run_id('rlq')
    run_dir = OUT / 'runs' / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) 探索采集（复用已落盘数据则跳过；否则并行采集，每 worker 落盘）
    from multiprocessing import Pool
    parts_exist = all((OUT / f'collect-{s}.npz').exists() for s in args.seeds)
    if args.reuse_data and parts_exist:
        print(f'=== 复用已落盘探索数据（{len(args.seeds)} seeds）===', flush=True)
        t0 = time.perf_counter()
    else:
        print(f'=== 探索采集（无老师偏置随机，{args.frames_per_seed} 帧/seed × {len(args.seeds)} seeds，并行 {args.workers}）===', flush=True)
        t0 = time.perf_counter()
        with Pool(args.workers) as pool:
            pool.starmap(collect_one,
                         [(s, args.frames_per_seed, uv, OUT) for s in args.seeds])
        print(f'  collect done: elapsed {time.perf_counter() - t0:.1f}s', flush=True)
    parts = [np.load(OUT / f'collect-{s}.npz') for s in args.seeds]
    counts_all = np.concatenate([p['counts'] for p in parts], axis=0).astype(np.float32)
    acts_all = np.concatenate([p['acts'] for p in parts], axis=0)
    rwds_all = np.concatenate([p['rwds'] for p in parts], axis=0)
    dones_all = np.concatenate([p['dones'] for p in parts], axis=0)
    # bug J：groups 标识 (seed, episode)，窗口与 FQI 边界都不跨组
    groups_all = []
    for i, p in enumerate(parts):
        if 'groups' in p.files:
            g = p['groups']
        else:
            print(f'WARN: {args.seeds[i]} 的旧数据无 groups 字段，按单 episode 处理 '
                  f'（建议重采该 seed）', flush=True)
            g = np.zeros(len(p['counts']), dtype=np.int64)
        groups_all.append(g.astype(np.int64) + i * 100000)
    groups_all = np.concatenate(groups_all)
    for p in parts:
        p.close()
    print(f'  collect done: {counts_all.shape} frames, {len(np.unique(groups_all))} episodes, '
          f'elapsed {time.perf_counter() - t0:.1f}s, '
          f'total_reward={rwds_all.sum():.1f} exit_frames={dones_all.sum()}', flush=True)

    # 2) 无监督特征选择 + 分组窗口构建（bug J：窗口不跨组）
    print('=== 无监督 topK 选择 + 特征构建 ===', flush=True)
    idx = select_topk_unsupervised(counts_all, args.topk)
    F = build_features_grouped(counts_all, groups_all, idx, WIN_FQI, DIFF_FQI).astype(np.float32)
    mu, sd = fit_stats(F)
    Fz = normalize(F, mu, sd)
    np.savez(run_dir / 'collect-data.npz',
             topk=idx, mu=mu, sd=sd,
             acts=acts_all, rwds=rwds_all, dones=dones_all, groups=groups_all)
    print(f'  features {Fz.shape} (topk={args.topk}, win={WIN_FQI}, diff={DIFF_FQI})', flush=True)

    # 3) 离线 Q 学习（bug J：FQI 边界由 groups 决定，不跨组转移）
    print(f'=== 离线 FQI（γ=0.99, {args.rounds} rounds, device={device}）===', flush=True)
    q, log = train_fqi(Fz, acts_all, rwds_all, dones_all, groups_all,
                       rounds=args.rounds, device=device)
    torch.save({'dim': Fz.shape[1], 'state': q.state_dict(), 'mu': mu, 'sd': sd,
                'topk': idx, 'log': log, 'run_id': rid}, run_dir / 'model.pt')
    write_provenance(run_dir, {'run_id': rid, 'script': 'corridor_rl_q.py',
                               'seeds': args.seeds, 'frames_per_seed': args.frames_per_seed,
                               'topk': args.topk, 'rounds': args.rounds, 'win': WIN_FQI,
                               'diff': DIFF_FQI, 'gamma': 0.99},
                     sources={'brain_graph': BRAIN_CLEAN,
                              'corridor_wad': Path(r'D:\fly\doomfly\doom\scenarios\corridor_avoid.wad'),
                              'script': Path(__file__)})
    json.dump({'run_id': rid, 'n_frames': len(F), 'topk': args.topk, 'rounds': args.rounds,
               'log': log, 'collect_total_reward': round(float(rwds_all.sum()), 1)},
              open(run_dir / 'train-log.json', 'w'), ensure_ascii=False, indent=1)
    print('  saved', run_dir / 'model.pt', flush=True)

    # 4) 闭环验证（独立 seed，bug L：结果写 runs/<rid>/ 不覆盖）
    print(f'=== 闭环验证（greedy, {args.eval_eps} eps × {args.eval_max_tics} tics, seed={args.eval_seed}）===', flush=True)
    q.eval()
    res = closed_loop(q, uv, args.eval_eps, args.eval_max_tics,
                      seed=args.eval_seed, idx=idx, mu=mu, sd=sd, device=device)
    res.update({'reference': {'teacher_exit_rate': 1.0, 'teacher_tics_mean': 565.2,
                              'teacher_wall_rate': 0.0007},
                'distill_student_exit_rate': 0.0})
    (OUT / 'latest.json').write_text(json.dumps(
        {'run_id': rid, 'model': str(run_dir / 'model.pt'),
         'closed_loop': str(run_dir / 'closed-loop-results.json'),
         'created': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print('closed-loop:', {k: v for k, v in res.items() if k != 'episodes'}, flush=True)
