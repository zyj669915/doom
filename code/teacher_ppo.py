"""Teacher agent v4: parameterized PPO on ViZDoom.

v4 changes (plan teacher-v4-plan.md):
  - obs modes: A gray+diff(1536) / B 4-frame stack(3072) / C ON-OFF energy(2304)
  - action modes: '12' joint 12-class (turn*fwd*fire) / 'md' multi-discrete heads (turn3,move3 incl back,fire2)
  - reward: dKills*10 - dHealth*health_w + dHITCOUNT*hit_bonus  (fire_bonus removed)
  - domain randomization: --rand 0/0.3/0.6 (brightness/contrast/shift on gray obs, training only)
  - cosine LR; unified eval 5 game seeds x 600 tics; hit_rate reported per ammo spent (not fire-frames)
"""
from pathlib import Path
import json, time, argparse, math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from doom.game import Game

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'doom' / 'teacher-v4'
OUT.mkdir(parents=True, exist_ok=True)

EVAL_SEEDS = [41027, 41028, 41029, 41030, 41031]
TURNS = [-6.0, 0.0, 6.0]
MOVES = [-20.0, 0.0, 20.0]   # back / stop / forward (md mode)
FWDS = [0.0, 20.0]            # 12-class mode
ATTK = [0, 1]
N_ACT12 = len(TURNS) * len(FWDS) * len(ATTK)  # 12
N_ACTMD = len(TURNS) * len(MOVES) * len(ATTK)  # 18

def index_to_action_12(i):
    at = i % 2; r = i // 2
    f = r % 2; t = r // 2
    return {'turn': TURNS[t], 'forward': FWDS[f], 'attack': at}

def index_to_action_md(i):
    f = i % 2; r = i // 2
    m = r % 3; t = r // 3
    return {'turn': TURNS[t], 'forward': MOVES[m], 'attack': f}

def downsample(rgb, rows=24, cols=32):
    g = rgb.reshape(rgb.shape[0]//rows, rows, rgb.shape[1]//cols, cols, 3).mean(axis=(0, 2, 4))
    return (g / 255.0).astype(np.float32)

def randomize(gray, rand):
    """Domain randomization on 24x32 gray (training only). rand in {0, 0.3, 0.6}."""
    if rand <= 0:
        return gray
    g = gray.copy()
    if np.random.rand() < 0.8:
        g = g + np.float32(np.random.uniform(-rand * 0.15, rand * 0.15))
    if np.random.rand() < 0.8:
        g = np.float32(np.clip((g - 0.5) * (1 + np.float32(np.random.uniform(-rand * 0.4, rand * 0.4))) + 0.5, 0, 1))
    if np.random.rand() < 0.6:
        sh = int(np.random.choice([-1, 1]) * np.random.randint(1, 2 + int(rand * 3)))
        g = np.roll(g, sh, axis=1)
    return np.clip(g, 0, 1).astype(np.float32)

class ObsBuilder:
    """Builds obs vector per mode. Keeps frame history (prev / 4-frame buffer)."""
    def __init__(self, mode='A', rand=0.0):
        self.mode = mode
        self.rand = rand
        self.prev = None
        self.frames = None
    def _gray(self, game):
        g = downsample(game.pixels())          # 2D (24,32)
        g = randomize(g, self.rand)            # randomize on 2D
        return g.ravel()
    def reset(self, game):
        cur = self._gray(game)
        self.prev = cur
        self.frames = deque([cur] * 4, maxlen=4)
        return self._vec(cur)
    def _vec(self, cur):
        if self.mode == 'A':
            return np.concatenate([cur, np.abs(cur - self.prev)]).astype(np.float32)
        if self.mode == 'B':
            return np.concatenate(list(self.frames)).astype(np.float32)
        if self.mode == 'C':  # ON-OFF: gray + brighten + darken
            d = cur - self.prev
            return np.concatenate([cur, np.maximum(d, 0), np.maximum(-d, 0)]).astype(np.float32)
        raise ValueError(self.mode)
    def step(self, game):
        """After one action: advance history, return next obs."""
        cur = self._gray(game)
        self.frames.append(cur)
        o = self._vec(cur)
        self.prev = cur
        return o

class Policy(nn.Module):
    def __init__(self, obs_dim=1536, action_mode='12'):
        super().__init__()
        self.action_mode = action_mode
        self.body = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(),
                                  nn.Linear(256, 128), nn.Tanh())
        if action_mode == '12':
            self.pol = nn.Linear(128, N_ACT12)
        else:
            self.pol_t = nn.Linear(128, len(TURNS))
            self.pol_m = nn.Linear(128, len(MOVES))
            self.pol_f = nn.Linear(128, len(ATTK))
        self.val = nn.Linear(128, 1)
    def forward(self, x):
        h = self.body(x)
        v = self.val(h).squeeze(-1)
        if self.action_mode == '12':
            return self.pol(h), v
        return (self.pol_t(h), self.pol_m(h), self.pol_f(h)), v
    def _cat_sample(self, logits_list):
        a, lp = [], 0.0
        for lt in logits_list:
            d = torch.distributions.Categorical(logits=lt)
            a.append(int(d.sample().item()))
            lp += d.log_prob(torch.tensor(a[-1]))
        idx = a[0] * (len(MOVES) * len(ATTK)) + a[1] * len(ATTK) + a[2]
        return idx, lp
    def act(self, x):
        out, v = self.forward(x)
        if self.action_mode == '12':
            d = torch.distributions.Categorical(logits=out)
            a = d.sample()
            return a, d.log_prob(a), v
        idx, lp = self._cat_sample(out)
        return torch.tensor(idx, dtype=torch.long), lp, v
    def evaluate(self, x, a):
        out, v = self.forward(x)
        if self.action_mode == '12':
            d = torch.distributions.Categorical(logits=out)
            return d.log_prob(a), d.entropy(), v
        # decode a (batch) into sub-indices
        at = a % 2; r = a // 2
        am = r % 3; at_ = r // 3
        dt = torch.distributions.Categorical(logits=out[0]); lm = out[1]; lf = out[2]
        dm = torch.distributions.Categorical(logits=lm)
        df = torch.distributions.Categorical(logits=lf)
        lp = dt.log_prob(at_) + dm.log_prob(am) + df.log_prob(at)
        ent = dt.entropy() + dm.entropy() + df.entropy()
        return lp, ent, v

def make_env(seed=41027, scenario='combat_survival'):
    return Game(seed=seed, scenario=scenario)

def idx_fire(i, mode):
    return (i % 2) if mode == '12' else (i % 2)

def run_rollout(policy, env, n_steps=4096, gamma=0.99, lam=0.95, health_w=5.0, hit_bonus=1.0,
                obs_mode='A', rand=0.0):
    obs_buf, act_buf, rew_buf, dones = [], [], [], []
    g = env
    ob = ObsBuilder(mode=obs_mode, rand=rand)
    if g.game.is_episode_finished():
        g.new_episode()
    o = ob.reset(g)
    st = g.observation()
    prev_k, prev_h, prev_hit = st['kills'], st['health'], st['hitcount']
    done = False
    fire_n, hit_n, kill_gain = 0, 0, 0
    for _ in range(n_steps):
        if done:
            g.new_episode()
            o = ob.reset(g)
            st = g.observation()
            prev_k, prev_h, prev_hit = st['kills'], st['health'], st['hitcount']
            done = False
        obs_buf.append(o)
        x = torch.from_numpy(o).unsqueeze(0)
        a, lp, v = policy.act(x)
        ai = int(a.item())
        if policy.action_mode == '12':
            fire_n += (ai % 2 == 1)
            act = index_to_action_12(ai)
        else:
            fire_n += (ai % 2 == 1)
            act = index_to_action_md(ai)
        act_buf.append(ai)
        rew = g.act(act)
        st = g.observation()
        k, h, hit = st['kills'], st['health'], st['hitcount']
        kill_gain += int(k - prev_k)
        hit_n += int(hit - prev_hit)
        r = (k - prev_k) * 10.0 - max(0, prev_h - h) * health_w + (hit - prev_hit) * hit_bonus
        rew_buf.append(r)
        prev_k, prev_h, prev_hit = k, h, hit
        done = st['finished']
        dones.append(done)
        if done:
            g.new_episode()
        o = ob.step(g)
    print(f'[rollout] fires={fire_n} hits={hit_n} kill_gain={kill_gain}', flush=True)
    return (np.stack(obs_buf), np.array(act_buf), np.array(rew_buf, dtype=np.float32),
            np.array(dones, dtype=np.float32))

def eval_policy(policy, seeds=EVAL_SEEDS, scenario='combat_survival', tics=600, obs_mode='A'):
    """Greedy evaluation, NO randomization. Reports per-seed metrics + aggregates.
    hit_rate_ammo = hits / ammo_spent (true per-shot hit rate, only seeds with positive ammo spend)."""
    per_seed = []
    dist = np.zeros(N_ACT12 if policy.action_mode == '12' else N_ACTMD, dtype=np.int64)
    for sd in seeds:
        env = make_env(seed=sd, scenario=scenario)
        env.new_episode()
        ob = ObsBuilder(mode=obs_mode, rand=0.0)
        o = ob.reset(env)
        st = env.observation()
        k0, h0, hit0, ammo0 = st['kills'], st['health'], st['hitcount'], st['ammo']
        done = False
        t = 0
        hits, fires, ammo_used, health_lost = 0, 0, 0, 0
        last_h, last_ammo = h0, ammo0
        while not done and t < tics:
            a = int(policy.act(torch.from_numpy(o).unsqueeze(0))[0].item())
            fires += (a % 2 == 1)
            dist[a] += 1
            if policy.action_mode == '12':
                env.act(index_to_action_12(a))
            else:
                env.act(index_to_action_md(a))
            t += 1
            st = env.observation()
            hits += int(st['hitcount'] - hit0); hit0 = st['hitcount']
            ammo_used += int(last_ammo - st['ammo']); last_ammo = st['ammo']
            health_lost += max(0, last_h - st['health']); last_h = st['health']
            done = st['finished']
            if not done:
                o = ob.step(env)
        per_seed.append({'seed': sd, 'kills': int(st['kills'] - k0), 'tics': t,
                         'health': int(st['health']), 'health_lost': health_lost,
                         'hits': hits, 'fires': fires, 'ammo_used': ammo_used})
        env.close()
    m = {k: float(np.mean([s[k] for s in per_seed])) for k in
         ['kills', 'tics', 'health', 'health_lost', 'hits', 'fires', 'ammo_used']}
    m['kills_std'] = float(np.std([s['kills'] for s in per_seed]))
    m['hit_rate_frames'] = m['hits'] / (m['fires'] + 1e-8)
    pos = [s for s in per_seed if s['ammo_used'] > 0]
    m['hit_rate_ammo'] = float(np.mean([s['hits'] / max(1, s['ammo_used']) for s in pos])) if pos else 0.0
    m['ammo_per_kill'] = m['ammo_used'] / (m['kills'] + 1e-8)
    m['action_dist'] = (dist / dist.sum()).tolist() if dist.sum() > 0 else []
    return {'metrics': m, 'per_seed': per_seed}

def train(policy, env, opt, steps_total=60000, n_steps=4096, epochs=10, clip=0.2,
          gamma=0.99, lam=0.95, lr=3e-4, lr_final=3e-5, health_w=5.0, hit_bonus=1.0,
          obs_mode='A', rand=0.0, eval_every=5000, t0=None, start_step=0, log0=None):
    log = list(log0) if log0 else []
    if t0 is None:
        t0 = time.time()
    step = start_step
    best_score = -1e18
    while step < steps_total:
        obs, acts, rews, dones = run_rollout(policy, env, n_steps, gamma, lam, health_w, hit_bonus,
                                             obs_mode, rand)
        with torch.no_grad():
            v = policy.forward(torch.from_numpy(obs))[1].numpy().ravel()
        gae = np.zeros_like(rews); adv = 0.0
        for i in reversed(range(len(rews))):
            v_next = (0.0 if dones[i] else v[i]) if i == len(rews) - 1 else (0.0 if dones[i] else v[i + 1])
            delta = rews[i] + gamma * v_next - v[i]
            adv = delta + gamma * lam * (0.0 if dones[i] else adv)
            gae[i] = adv
        returns = gae + v
        adv_n = (gae - gae.mean()) / (gae.std() + 1e-8)
        for _ in range(epochs):
            idx = np.arange(len(obs))
            np.random.shuffle(idx)
            for b in range(0, len(obs), 512):
                bi = idx[b:b + 512]
                ob = torch.from_numpy(obs[bi]); ac = torch.from_numpy(acts[bi])
                logp, ent, val = policy.evaluate(ob, ac)
                with torch.no_grad():
                    old_logp = policy.evaluate(ob, ac)[0]
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * torch.from_numpy(adv_n[bi])
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * torch.from_numpy(adv_n[bi])
                ploss = -torch.min(surr1, surr2).mean()
                vloss = F.mse_loss(val, torch.from_numpy(returns[bi]))
                loss = ploss + 0.5 * vloss - 0.02 * ent.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        step += n_steps
        frac = min(1.0, step / steps_total)
        opt.param_groups[0]['lr'] = lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * frac))
        if step % eval_every < n_steps or step >= steps_total:
            res = eval_policy(policy, scenario=env.scenario, tics=600, obs_mode=obs_mode)
            m = res['metrics']
            log.append({'step': step, 'kills_mean': m['kills'], 'kills_std': m['kills_std'],
                        'tics_mean': m['tics'], 'health_mean': m['health'],
                        'health_lost_mean': m['health_lost'],
                        'hit_rate_ammo': m['hit_rate_ammo'],
                        'ammo_per_kill': m['ammo_per_kill'],
                        'elapsed_s': round(time.time() - t0, 1)})
            print(f"step {step}: kills={m['kills']:.2f}±{m['kills_std']:.2f} tics={m['tics']:.0f} "
                  f"health={m['health']:.1f} hit/ammo={m['hit_rate_ammo']*100:.0f}%", flush=True)
            ckpt = {'policy': policy.state_dict(), 'log': log, 'cfg': {
                'obs_mode': obs_mode, 'action_mode': policy.action_mode, 'rand': rand,
                'health_w': health_w, 'hit_bonus': hit_bonus, 'steps_total': steps_total}}
            torch.save(ckpt, OUT / 'teacher.pt')
            torch.save(ckpt, OUT / f'teacher-{step}.pt')
            score = m['kills'] + m['health'] / 20.0
            if score > best_score:
                best_score = score
                torch.save(ckpt, OUT / 'best.pt')
            json.dump(log, open(OUT / 'teacher-log.json', 'w'), ensure_ascii=False, indent=1)
    return log

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--seed', type=int, default=41027)
    ap.add_argument('--resume', default=None, help='dir of checkpoint to resume from (policy+cfg+log)')
    ap.add_argument('--scenario', default='combat_survival')
    ap.add_argument('--out', default=None)
    ap.add_argument('--obs', default='A', choices=['A', 'B', 'C'])
    ap.add_argument('--action', default='12', choices=['12', 'md'])
    ap.add_argument('--rand', type=float, default=0.0)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lr-final', type=float, default=3e-5)
    ap.add_argument('--health-w', type=float, default=5.0)
    ap.add_argument('--hit-bonus', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=5000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    if args.out:
        OUT = Path(args.out)
        OUT.mkdir(parents=True, exist_ok=True)
    obs_dim = {'A': 24 * 32 * 2, 'B': 24 * 32 * 4, 'C': 24 * 32 * 3}[args.obs]
    env = make_env(seed=args.seed, scenario=args.scenario)
    policy = Policy(obs_dim=obs_dim, action_mode=args.action)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    start_step = 0
    log = []
    if args.resume:
        ckpt = torch.load(Path(args.resume) / 'teacher.pt', map_location='cpu')
        policy.load_state_dict(ckpt['policy'])
        start_step = ckpt['log'][-1]['step'] if ckpt.get('log') else 0
        # restore lr schedule state from checkpoint config
        cfg = ckpt.get('cfg', {})
        obs_dim = {'A': 24 * 32 * 2, 'B': 24 * 32 * 4, 'C': 24 * 32 * 3}[cfg.get('obs_mode', 'A')]
        args.obs = cfg.get('obs_mode', args.obs)
        args.action = cfg.get('action_mode', args.action)
        args.health_w = cfg.get('health_w', args.health_w)
        args.hit_bonus = cfg.get('hit_bonus', args.hit_bonus)
        args.rand = cfg.get('rand', args.rand)
        log = ckpt.get('log', [])
        print(f'[resume] from step {start_step} (obs={args.obs} action={args.action} '
              f'hw={args.health_w} bonus={args.hit_bonus} rand={args.rand})', flush=True)
    # NOTE: eval log kept by train(); pass preloaded log if we want continuity of teacher-log.json
    log = train(policy, env, opt, steps_total=args.steps, lr=args.lr, lr_final=args.lr_final,
                health_w=args.health_w, hit_bonus=args.hit_bonus, obs_mode=args.obs,
                rand=args.rand, eval_every=args.eval_every, t0=t0, start_step=start_step,
                log0=log)
    print('done in', round(time.time() - t0, 1), 's')

if __name__ == '__main__':
    main()
