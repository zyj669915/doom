"""Teacher agent: mini-PPO on ViZDoom combat_survival.
A deliberately small (MLP + downsampled greyscale) policy — enough to serve as a
'teacher' for distilling into the fly brain. Records kills/health curve per eval.
"""
from pathlib import Path
import json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from doom.game import Game

ROOT = Path(r'D:\fly\doomfly')
OUT = ROOT / 'outputs' / 'doom' / 'teacher-v1'
OUT.mkdir(parents=True, exist_ok=True)

TURNS = [-6.0, 0.0, 6.0]
FWDS = [0.0, 20.0]
ATTK = [0, 1]
N_ACT = len(TURNS) * len(FWDS) * len(ATTK)  # 12

def act_index(a):
    t, f, at = a
    return (int(np.searchsorted(TURNS, t)) * len(FWDS) + int(np.searchsorted(FWDS, f))) * len(ATTK) + int(at)

def index_to_action(i):
    at = i % 2; r = i // 2
    f = r % 2; t = r // 2
    return {'turn': TURNS[t], 'forward': FWDS[f], 'attack': at}

def downsample(rgb, rows=24, cols=32):
    # 640x480 -> rows x cols greyscale by block-mean
    g = rgb.reshape(rgb.shape[0]//rows, rows, rgb.shape[1]//cols, cols, 3).mean(axis=(0, 2, 4))
    return (g / 255.0).astype(np.float32)

class Policy(nn.Module):
    def __init__(self, obs_dim=24*32, n_act=N_ACT):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(),
                                  nn.Linear(256, 128), nn.Tanh())
        self.pol = nn.Linear(128, n_act)
        self.val = nn.Linear(128, 1)
    def forward(self, x):
        h = self.body(x)
        return self.pol(h), self.val(h).squeeze(-1)
    def act(self, x):
        logits, v = self.forward(x)
        d = torch.distributions.Categorical(logits=logits)
        a = d.sample()
        return a, d.log_prob(a), v
    def evaluate(self, x, a):
        logits, v = self.forward(x)
        d = torch.distributions.Categorical(logits=logits)
        return d.log_prob(a), d.entropy(), v

def make_env(seed=41027):
    g = Game(seed=seed, scenario='combat_survival')
    return g

def obs_vec(game):
    return downsample(game.pixels()).ravel()

def run_rollout(policy, env, n_steps=1024, gamma=0.99, lam=0.95):
    obs_buf, act_buf, rew_buf, dones = [], [], [], []
    g = env
    if g.game.is_episode_finished():
        g.new_episode()
    o = obs_vec(g)
    prev_k = g.observation()['kills']; prev_h = g.observation()['health']
    done = False
    for _ in range(n_steps):
        if done:
            g.new_episode(); o = obs_vec(g)
            prev_k = g.observation()['kills']; prev_h = g.observation()['health']
            done = False
        obs_buf.append(o)
        x = torch.from_numpy(o).unsqueeze(0)
        a, lp, v = policy.act(x)
        act_buf.append(int(a.item()))
        # step env
        rew = g.act(index_to_action(int(a.item())))
        k = g.observation()['kills']; h = g.observation()['health']
        r = (k - prev_k) * 10.0 - max(0, prev_h - h) * 1.0
        rew_buf.append(r)
        prev_k, prev_h = k, h
        done = g.observation()['finished']
        dones.append(done)
        if done:
            g.new_episode()
        o = obs_vec(g)
    return (np.stack(obs_buf), np.array(act_buf), np.array(rew_buf, dtype=np.float32),
            np.array(dones, dtype=np.float32))

def train(policy, env, opt, steps_total=30000, n_steps=1024, epochs=6, clip=0.2, gamma=0.99, lam=0.95, lr=3e-4):
    log = []
    step = 0
    eval_every = 3000
    while step < steps_total:
        obs, acts, rews, dones = run_rollout(policy, env, n_steps, gamma, lam)
        # GAE
        with torch.no_grad():
            v = policy.forward(torch.from_numpy(obs))[1].numpy().ravel()
        gae = np.zeros_like(rews); adv = 0.0; ret = 0.0
        next_v = 0.0
        for i in reversed(range(len(rews))):
            if i == len(rews)-1:
                v_next = 0.0 if dones[i] else v[i]
            else:
                v_next = 0.0 if dones[i] else v[i+1]
            delta = rews[i] + gamma * v_next - v[i]
            adv = delta + gamma * lam * (0.0 if dones[i] else adv)
            gae[i] = adv
        returns = gae + v
        adv_n = (gae - gae.mean()) / (gae.std() + 1e-8)
        for _ in range(epochs):
            idx = np.arange(len(obs))
            np.random.shuffle(idx)
            for b in range(0, len(obs), 256):
                bi = idx[b:b+256]
                ob = torch.from_numpy(obs[bi]); ac = torch.from_numpy(acts[bi])
                logp, ent, val = policy.evaluate(ob, ac)
                with torch.no_grad():
                    old_logp = policy.evaluate(ob, ac)[0]
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * torch.from_numpy(adv_n[bi])
                surr2 = torch.clamp(ratio, 1-clip, 1+clip) * torch.from_numpy(adv_n[bi])
                ploss = -torch.min(surr1, surr2).mean()
                vloss = F.mse_loss(val, torch.from_numpy(returns[bi]))
                loss = ploss + 0.5 * vloss - 0.01 * ent.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        step += n_steps
        # eval
        k = eval_policy(policy, env, 3)
        log.append({'step': step, 'kills_mean': float(k), 'elapsed_s': round(time.time()-t0, 1)})
        print(f'step {step}: kills_mean={k:.2f}', flush=True)
        torch.save({'policy': policy.state_dict(), 'log': log}, OUT/'teacher.pt')
        json.dump(log, open(OUT/'teacher-log.json', 'w'), ensure_ascii=False, indent=1)
    return log

def eval_policy(policy, env, n_ep=3):
    total = 0
    for _ in range(n_ep):
        env.new_episode(); o = obs_vec(env)
        prev_k = env.observation()['kills']; prev_h = env.observation()['health']
        done = False
        while not done:
            a = int(policy.act(torch.from_numpy(o).unsqueeze(0))[0].item())
            env.act(index_to_action(a))
            done = env.observation()['finished']
            if env.tick > 400: break
            if not done:
                o = obs_vec(env)
        total += env.observation()['kills']
    return total / n_ep

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--lr', type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(41027)
    np.random.seed(41027)
    t0 = time.time()
    env = make_env()
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    log = train(policy, env, opt, steps_total=args.steps)
    print('done in', round(time.time()-t0, 1), 's')
