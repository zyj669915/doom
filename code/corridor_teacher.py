# -*- coding: utf-8 -*-
"""走廊避障 PPO 老师训练（Windows 重建，v3 版，bug I/L 修复）。

观测：24x32 灰度 + 帧差（1536）+ 可选 ANGLE cos/sin（2）= 1536 或 1538 维。
      —— 报告 §5.1 I：ANGLE 是老师独享、学生不可见的引擎内部状态。
         修复：use_angle 可配置，输出「纯像素」与「带 ANGLE」两套老师
         checkpoint（§8.5 第 5 条），主训练用纯像素，ANGLE 版只作诊断上限。
动作：转向 {-15, 0, +15} x 前进 {0, 20} = 6 类（±15°/tick，90° 弯只需 6 ticks）。
奖励：撞墙 -0.15；进度 shaping +0.01*(d_prev-d_cur)；出口 +10。
训练：PPO（GAE/γ0.99/clip0.2/ent0.01/4 epochs/batch64/T256/cosine lr 3e-4→3e-5）。

修复记录：
  * bug I：ObsBuilder(use_angle=...)，train(--use-angle 0|1)，checkpoint 带 meta；
            load_teacher() 兼容旧版裸 state_dict（视为 v6 带 ANGLE）。
  * bug L：每次保存 {state, meta}，meta 含 use_angle/obs_dim/scenario/WAD sha/训练参数。
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import vizdoom as vzd

DOOMFLY = Path(r'D:\fly\doomfly')
sys.path.insert(0, str(DOOMFLY))
from doom.game import Game  # noqa: E402

OUT = Path(r'D:\fly\experiment\code\outputs\doom\corridor-teacher-v6')
SCENARIO = 'corridor_avoid'
WAD_SHA256 = 'f1afb773b7275854100691e7e99f9b80c1318da87ead459c25ba9ed9669c9efa'
OBS_GRAY_DIM = 24 * 32 * 2          # 灰度 + 帧差
ANGLE_DIM = 2
N_TURN, N_FWD = 3, 2

# 奖励（v6：恢复弱惩罚 0.15 —— 640×480 下云 VM 同配置 argmax 100% 成功）
WALL_PENALTY = -0.15
PROGRESS_GAIN = 0.01
EXIT_REWARD = 10.0
MAX_TICS = 2000
EXIT_X_MIN, EXIT_Y_MIN = 1792, 2048


def obs_dim(use_angle):
    return OBS_GRAY_DIM + (ANGLE_DIM if use_angle else 0)


def dist(x, y):
    """走廊中线到出口的剩余距离（Doom units）。起点 (128,128)=3456，出口=0。"""
    if y < 256:            # 入口段
        return (1792 - x) + (2048 - 256)
    if x < 1792:           # 转角段
        return (2048 - y) + (1792 - x)
    return 2048 - y        # 出口段


def at_exit(x, y):
    return x >= EXIT_X_MIN and y >= EXIT_Y_MIN


class ObsBuilder:
    """24x32 gray + frame-diff (1536 dims) + 可选朝向 cos/sin（bug I 修复）。"""
    def __init__(self, use_angle=True):
        self.prev = None
        self.use_angle = bool(use_angle)

    @staticmethod
    def _gray(rgb):
        p = rgb.astype(np.float32) / 255.0
        p = np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)
        y = p @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        h, w = y.shape[:2]
        g = y.reshape(24, h // 24, 32, w // 32).mean(axis=(1, 3))  # (24,32)
        return g

    @staticmethod
    def _angle(game):
        a = float(game.game.get_game_variable(vzd.GameVariable.ANGLE))
        rad = math.radians(a)
        return np.array([[math.cos(rad), math.sin(rad)]], dtype=np.float32)

    def _base(self, g):
        return np.concatenate([g.ravel(), np.zeros_like(g).ravel()], axis=0)

    def reset(self, game):
        rgb = game.pixels()
        g = self._gray(rgb)
        self.prev = g
        parts = [self._base(g)]
        if self.use_angle:
            parts.append(self._angle(game).ravel())
        return np.concatenate(parts, axis=0).astype(np.float32)

    def step(self, game):
        rgb = game.pixels()
        g = self._gray(rgb)
        d = g - self.prev
        self.prev = g
        parts = [np.concatenate([g.ravel(), d.ravel()], axis=0)]
        if self.use_angle:
            parts.append(self._angle(game).ravel())
        return np.concatenate(parts, axis=0).astype(np.float32)


class Policy(nn.Module):
    """v4：动作解耦双头 —— turn {−15,0,15} × forward {0,20} 独立输出。

    组合动作下 PPO 会把"直行"概率推至最高、转弯概率被边缘化（采样能过、
    argmax 必死）；双头让转向信号独立优化，对应果蝇下行运动神经元。
    """
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.pi_turn = nn.Linear(hidden, N_TURN)
        self.pi_fwd = nn.Linear(hidden, N_FWD)
        self.v = nn.Linear(hidden, 1)

    def act(self, obs, greedy=False):
        h = self.trunk(obs)
        lt, lf = self.pi_turn(h), self.pi_fwd(h)
        if greedy:
            return lt.argmax(-1), lf.argmax(-1), None, None
        dt = torch.distributions.Categorical(logits=lt)
        df = torch.distributions.Categorical(logits=lf)
        at, af = dt.sample(), df.sample()
        return at, af, dt.log_prob(at) + df.log_prob(af), self.v(h).squeeze(-1)

    def evaluate(self, obs, at, af):
        h = self.trunk(obs)
        dt = torch.distributions.Categorical(logits=self.pi_turn(h))
        df = torch.distributions.Categorical(logits=self.pi_fwd(h))
        return (dt.log_prob(at) + df.log_prob(af),
                dt.entropy().mean() + df.entropy().mean(),
                self.v(h).squeeze(-1))


def head_to_action(turn_i, fwd_i):
    return {'turn': [-15, 0, 15][int(turn_i)], 'forward': [0, 20][int(fwd_i)], 'attack': 0}


def load_teacher(path):
    """加载老师 checkpoint，返回 (policy, meta)。

    兼容新格式 {'state', 'meta'} 与旧版裸 state_dict（按 v6 带 ANGLE 处理）。
    """
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'state' in obj and 'meta' in obj:
        meta = dict(obj['meta'])
        state = obj['state']
    else:
        meta = {'use_angle': True, 'obs_dim': obs_dim(True), 'legacy': True,
                'wad_sha256': WAD_SHA256, 'scenario': SCENARIO}
        state = obj
    meta.setdefault('obs_dim', obs_dim(meta.get('use_angle', True)))
    policy = Policy(obs_dim=int(meta['obs_dim']))
    policy.load_state_dict(state)
    policy.eval()
    return policy, meta


def make_game(seed):
    g = Game(seed=seed, scenario=SCENARIO)
    if g.assets['sha256'].get('corridor_avoid.wad') != WAD_SHA256:
        raise RuntimeError('corridor WAD checksum mismatch')
    return g


def train(total_steps, seed, out, resume=False, use_angle=True):
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tag = 'angle' if use_angle else 'pixel'
    d_in = obs_dim(use_angle)
    game = make_game(seed)
    policy = Policy(d_in)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    n_act = N_TURN * N_FWD
    T = 256
    gamma, lam, clip, ent_coef, vf_coef = 0.99, 0.95, 0.2, 0.01, 0.5
    epochs, batch = 4, 64

    def lr_at(step):
        frac = min(step / total_steps, 1.0)
        return 3e-4 * (0.1 ** (0.5 * (1 - math.cos(math.pi * frac))))

    step = 0
    train_log, ep_log = [], []
    ckpt = None
    final_path = out / f'teacher-corridor-{tag}-final.pt'
    if resume and final_path.exists():
        pol, meta = load_teacher(final_path)
        policy.load_state_dict(pol.state_dict())
        # 从旧日志恢复步数
        for fn, lst in (('train-log.jsonl', train_log), ('episode-log.jsonl', ep_log)):
            fp = out / fn
            if fp.exists():
                lst.extend(json.loads(l) for l in fp.read_text().splitlines() if l.strip())
        if train_log:
            step = int(train_log[-1]['step'])
        print(f'RESUME from {final_path.name} (tag={tag}, step={step})', flush=True)

    ep_rew, ep_tics, ep_hits, ep_exit = 0.0, 0, 0, False
    recent_exit, recent_wall, recent_rew = [], [], []

    obs_buf = np.zeros((T, d_in), dtype=np.float32)
    turn_buf = np.zeros(T, dtype=np.int64)
    fwd_buf = np.zeros(T, dtype=np.int64)
    logp_buf = np.zeros(T, dtype=np.float32)
    rew_buf = np.zeros(T, dtype=np.float32)
    val_buf = np.zeros(T, dtype=np.float32)
    done_buf = np.zeros(T, dtype=bool)

    ob = ObsBuilder(use_angle=use_angle)
    obs = ob.reset(game)
    buf_i = 0
    t0 = time.time()

    def save_checkpoint(step):
        meta = {
            'use_angle': use_angle, 'obs_dim': d_in, 'scenario': SCENARIO,
            'wad_sha256': WAD_SHA256, 'version': 2,
            'total_steps': total_steps, 'seed': seed,
            'saved_step': step,
            'created': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        }
        torch.save({'state': policy.state_dict(), 'meta': meta}, final_path)
        (out / f'teacher-corridor-{tag}-final.meta.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
        # 向后兼容别名：最新的那份同时写 teacher-corridor-final.pt
        shutil.copyfile(final_path, out / 'teacher-corridor-final.pt')
        (out / 'teacher-corridor-final.meta.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
        torch.save({'state': policy.state_dict(), 'meta': meta},
                   out / f'teacher-corridor-{tag}-{step}.pt')

    while step < total_steps:
        at, af, logp, val = policy.act(torch.from_numpy(obs).float().unsqueeze(0))
        act = head_to_action(at.item(), af.item())
        x0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
        y0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
        r0 = game.act(act)
        x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
        y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
        moved = math.hypot(x1 - x0, y1 - y0)
        d_prev, d_cur = dist(x0, y0), dist(x1, y1)
        r = r0 + PROGRESS_GAIN * (d_prev - d_cur)
        if act['forward'] > 0 and moved < 0.5:
            r += WALL_PENALTY
            ep_hits += 1
        if at_exit(x1, y1):
            r += EXIT_REWARD
            ep_exit = True

        obs_buf[buf_i] = obs
        turn_buf[buf_i] = at.item()
        fwd_buf[buf_i] = af.item()
        logp_buf[buf_i] = logp.item()
        rew_buf[buf_i] = r
        val_buf[buf_i] = val.item()
        done = game.game.is_episode_finished() or at_exit(x1, y1)
        done_buf[buf_i] = done
        buf_i += 1
        ep_rew += r
        ep_tics += 1
        step += 1

        if done or ep_tics >= MAX_TICS:
            recent_exit.append(1 if ep_exit else 0)
            recent_wall.append(ep_hits / max(ep_tics, 1))
            recent_rew.append(ep_rew)
            if len(recent_exit) > 20:
                recent_exit.pop(0)
                recent_wall.pop(0)
                recent_rew.pop(0)
            ep_log.append({'episode': len(ep_log) + 1, 'tics': ep_tics,
                           'reward': round(ep_rew, 3), 'exit': bool(ep_exit),
                           'wall_hits': ep_hits})
            game.new_episode()
            obs = ob.reset(game)
            ep_rew, ep_tics, ep_hits, ep_exit = 0.0, 0, 0, False
        else:
            obs = ob.step(game)

        if buf_i == T:
            # GAE
            obs_t = torch.from_numpy(obs_buf)
            turn_t = torch.from_numpy(turn_buf)
            fwd_t = torch.from_numpy(fwd_buf)
            _, _, _, last_val = policy.act(torch.from_numpy(obs).float().unsqueeze(0))
            adv = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for i in reversed(range(T)):
                next_val = last_val.item() if i == T - 1 else val_buf[i + 1]
                delta = rew_buf[i] + gamma * next_val * (0.0 if done_buf[i] else 1.0) - val_buf[i]
                gae = delta + gamma * lam * (0.0 if done_buf[i] else 1.0) * gae
                adv[i] = gae
            ret = adv + val_buf
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(epochs):
                idx = np.random.permutation(T)
                for s in range(0, T, batch):
                    b = idx[s:s + batch]
                    logp_b, ent_b, v_b = policy.evaluate(obs_t[b], turn_t[b], fwd_t[b])
                    ratio = torch.exp(logp_b - torch.from_numpy(logp_buf[b]))
                    adv_b = torch.from_numpy(adv[b])
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b
                    loss = (-torch.min(surr1, surr2).mean()
                            + vf_coef * (v_b - torch.from_numpy(ret[b])).pow(2).mean()
                            - ent_coef * ent_b)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            buf_i = 0

        if step % 10240 == 0:
            rec = {'step': step,
                   'exit_rate': round(float(np.mean(recent_exit)), 4) if recent_exit else 0.0,
                   'wall_rate': round(float(np.mean(recent_wall)), 4) if recent_wall else 0.0,
                   'mean_r': round(float(np.mean(recent_rew)), 4) if recent_rew else 0.0,
                   'lr': round(opt.param_groups[0]['lr'], 8),
                   'eps': len(ep_log),
                   'elapsed': round(time.time() - t0, 1)}
            train_log.append(rec)
            print(f"step {rec['step']}: rew={rec['mean_r']} exit_rate={rec['exit_rate']} "
                  f"wall_rate={rec['wall_rate']} eps={rec['eps']} lr={rec['lr']:.2e}", flush=True)
            (out / 'train-log.jsonl').write_text('\n'.join(json.dumps(x) for x in train_log) + '\n')
            (out / 'episode-log.jsonl').write_text('\n'.join(json.dumps(x) for x in ep_log) + '\n')
            save_checkpoint(step)

    print('DONE', flush=True)
    (out / 'train-log.jsonl').write_text('\n'.join(json.dumps(x) for x in train_log) + '\n')
    (out / 'episode-log.jsonl').write_text('\n'.join(json.dumps(x) for x in ep_log) + '\n')
    save_checkpoint(step)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--total-steps', type=int, default=200000)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--use-angle', type=int, default=1, choices=[0, 1],
                   help='1=带 ANGLE 特权输入（诊断上限）；0=纯像素（学生主训练）')
    args = p.parse_args()
    train(args.total_steps, args.seed, OUT, resume=args.resume,
          use_angle=bool(args.use_angle))
