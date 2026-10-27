"""v2 DoubleDQN/PPO；discount 始终是完整逐转移折扣，绝不额外乘 gamma。"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import ActorCritic, ModelSpec, QNetwork, observation, positive_int


def bounded(name, value, low=0., high=1.):
    value = float(value)
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} 必须在 [{low}, {high}] 内")
    return value


def _array(value):
    return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.array(value, copy=True)


def double_dqn_targets(rewards, next_online_q, next_target_q, discounts):
    """reward + discount * Q_target(next_obs, argmax Q_online)。"""
    inputs = (rewards, next_online_q, next_target_q, discounts)
    reference = next((x for x in inputs if isinstance(x, torch.Tensor)), None)
    device = reference.device if reference is not None else "cpu"
    with torch.no_grad():
        r, online, target, d = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in inputs]
        if online.ndim != 2 or target.shape != online.shape or online.shape[1] < 1:
            raise ValueError("Q 必须为同形状 (N,actions)")
        if r.shape != (len(online),) or d.shape != r.shape:
            raise ValueError("rewards/discounts 必须为 (N,)")
        if not all(torch.isfinite(x).all().item() for x in (r, online, target, d)) or torch.any((d < 0) | (d > 1)).item():
            raise ValueError("target 输入必须有限且 discount 在 [0,1]")
        result = r + d * target.gather(1, online.argmax(1, keepdim=True)).squeeze(1)
    return result if reference is not None else result.numpy()


def compute_gae(rewards, values, next_values, discounts, episode_end=None, *, gae_lambda=.95):
    """动态 discount 的 GAE；episode_end 切断递推但保留 truncation bootstrap。"""
    lam = bounded("gae_lambda", gae_lambda)
    r, v, nv, d = [np.asarray(_array(x), dtype=np.float64) for x in (rewards, values, next_values, discounts)]
    if r.ndim not in (1, 2) or not len(r) or any(x.shape != r.shape for x in (v, nv, d)):
        raise ValueError("GAE 输入必须是同形状非空 (T,) 或 (T,N)")
    if not all(np.isfinite(x).all() for x in (r, v, nv, d)) or np.any((d < 0) | (d > 1)):
        raise ValueError("GAE 输入必须有限且 discount 在 [0,1]")
    end = np.zeros_like(r, dtype=bool) if episode_end is None else np.asarray(_array(episode_end), dtype=bool)
    if end.shape != r.shape:
        raise ValueError("episode_end 形状错误")
    delta = r + d * nv - v
    advantages = np.empty_like(r)
    carry = np.zeros_like(r[0])
    for t in range(len(r)-1, -1, -1):
        carry = delta[t] + d[t] * lam * np.where(end[t], 0., carry)
        advantages[t] = carry
    return advantages.astype(np.float32), (advantages + v).astype(np.float32)


class ReplayBuffer:
    """惰性分配槽位，图像保存 uint8，不预先分配两个巨大图像数组。"""
    def __init__(self, spec, capacity=20_000, seed=0):
        self.spec = ModelSpec.coerce(spec)
        self.capacity = positive_int("capacity", capacity)
        self.position = 0
        self.storage = []
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.storage)

    def add(self, obs, action, reward, next_obs, discount, *, terminated=False,
            truncated=False, final_observation=None):
        obs = observation(self.spec, obs)
        nxt = observation(self.spec, next_obs if final_observation is None else final_observation)
        if isinstance(action, (bool, np.bool_)) or int(action) != action or not 0 <= action < self.spec.actions:
            raise ValueError("action 超出动作空间")
        if not np.isfinite(reward):
            raise ValueError("reward 必须有限")
        discount = bounded("discount", discount)
        if terminated and discount != 0:
            raise ValueError("terminated 转移 discount 必须为 0")
        item = dict(obs=obs, actions=int(action), rewards=float(reward), next_obs=nxt, discounts=discount)
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, rng=None):
        batch_size = positive_int("batch_size", batch_size)
        if batch_size > len(self):
            raise ValueError("replay 数据不足")
        indices = (self.rng if rng is None else rng).choice(len(self), batch_size, replace=False)
        rows = [self.storage[i] for i in indices]
        return {"obs": _stack([x["obs"] for x in rows]), "next_obs": _stack([x["next_obs"] for x in rows]),
                "actions": np.array([x["actions"] for x in rows], dtype=np.int64),
                **{key: np.array([x[key] for x in rows], dtype=np.float32) for key in ("rewards", "discounts")}}

    def state_dict(self):
        rows = [{**row, "obs": {k: torch.from_numpy(v.copy()) for k, v in row["obs"].items()},
                 "next_obs": {k: torch.from_numpy(v.copy()) for k, v in row["next_obs"].items()}} for row in self.storage]
        return dict(spec=self.spec.to_dict(), capacity=self.capacity, position=self.position,
                    rows=rows, rng=deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, data):
        if ModelSpec.coerce(data["spec"]) != self.spec or data["capacity"] != self.capacity:
            raise ValueError("replay spec/capacity 不匹配")
        rows, pos = data["rows"], data["position"]
        if len(rows) > self.capacity or not 0 <= pos < self.capacity or (len(rows) < self.capacity and pos != len(rows)):
            raise ValueError("replay 游标错误")
        fresh = ReplayBuffer(self.spec, self.capacity)
        for row in rows:
            fresh.add(row["obs"], row["actions"], row["rewards"], row["next_obs"], row["discounts"])
        fresh.rng.bit_generator.state = deepcopy(data["rng"])
        self.storage, self.position, self.rng = fresh.storage, pos, fresh.rng
        return self


def _stack(rows):
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


@dataclass(frozen=True)
class DQNConfig:
    device: str = "cpu"
    seed: int = 0
    lr: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 20_000
    learning_starts: int = 256
    target_every: int = 500
    max_grad_norm: float = 10.

    def __post_init__(self):
        _common_config(self)
        for key in ("batch_size", "replay_capacity", "target_every"):
            object.__setattr__(self, key, positive_int(key, getattr(self, key)))
        if isinstance(self.learning_starts, bool) or int(self.learning_starts) != self.learning_starts or self.learning_starts < 0:
            raise ValueError("learning_starts 必须非负整数")
        object.__setattr__(self, "learning_starts", int(self.learning_starts))
        if max(self.batch_size, self.learning_starts) > self.replay_capacity:
            raise ValueError("replay 必须容纳 batch 和 warmup")


@dataclass(frozen=True)
class PPOConfig:
    device: str = "cpu"
    seed: int = 0
    lr: float = 3e-4
    gae_lambda: float = .95
    epochs: int = 4
    minibatch_size: int = 64
    clip_ratio: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float | None = .03
    normalize_advantage: bool = True

    def __post_init__(self):
        _common_config(self)
        object.__setattr__(self, "epochs", positive_int("epochs", self.epochs))
        object.__setattr__(self, "minibatch_size", positive_int("minibatch_size", self.minibatch_size))
        object.__setattr__(self, "gae_lambda", bounded("gae_lambda", self.gae_lambda))
        object.__setattr__(self, "clip_ratio", bounded("clip_ratio", self.clip_ratio))
        object.__setattr__(self, "normalize_advantage", bool(self.normalize_advantage))
        for key in ("entropy_coef", "value_coef"):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} 必须有限非负")
            object.__setattr__(self, key, float(value))
        if self.target_kl is not None and (not np.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl 必须有限正数或 None")
        if self.target_kl is not None:
            object.__setattr__(self, "target_kl", float(self.target_kl))


def _common_config(c):
    for key in ("lr", "max_grad_norm"):
        if not np.isfinite(getattr(c, key)) or getattr(c, key) <= 0:
            raise ValueError(f"{key} 必须有限正数")
        object.__setattr__(c, key, float(getattr(c, key)))
    if isinstance(c.seed, (bool, np.bool_)) or int(c.seed) != c.seed or c.seed < 0:
        raise ValueError("seed 必须非负整数")
    object.__setattr__(c, "seed", int(c.seed))
    object.__setattr__(c, "device", str(c.device))


def _config(cls, config, kwargs):
    values = {} if config is None else asdict(config) if isinstance(config, cls) else dict(config)
    return cls(**{**values, **kwargs})


class _Learner:
    def _init(self, spec, config, model_class):
        self.spec, self.config = ModelSpec.coerce(spec), config
        self.device = torch.device(config.device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.model = model_class(self.spec).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.rng = np.random.default_rng(config.seed)
        self.torch_rng = torch.Generator(device="cpu").manual_seed(config.seed)
        self.updates = 0

    def state_dict(self, *, include_optimizer=True, include_rng=True, include_replay=True):
        data = dict(version=1, kind=type(self).__name__, spec=self.spec.to_dict(),
                    config=asdict(self.config), model=self.model.state_dict(), updates=self.updates)
        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()
        if include_rng:
            data.update(rng=self.rng.bit_generator.state, torch_rng=self.torch_rng.get_state())
        if hasattr(self, "target"):
            data["target"] = self.target.state_dict()
        data = deepcopy(data)
        if hasattr(self, "replay") and include_replay:
            data["replay"] = self.replay.state_dict()
        return data

    def load_state_dict(self, data, *, strict=True, restore_optimizer=True, restore_rng=True, restore_replay=True):
        if data.get("version") != 1 or data.get("kind") != type(self).__name__:
            raise ValueError("不受支持的 learner checkpoint")
        if ModelSpec.coerce(data["spec"]) != self.spec:
            raise ValueError("checkpoint ModelSpec 不匹配")
        expected, actual = asdict(self.config), dict(data["config"])
        expected.pop("device"); actual.pop("device")
        if strict and expected != actual:
            raise ValueError("严格恢复要求除 device 外 config 相同")
        self.model.load_state_dict(data["model"], strict=strict)
        if hasattr(self, "target"):
            self.target.load_state_dict(data["target"], strict=strict)
            if restore_replay:
                self.replay.load_state_dict(data["replay"])
        if restore_optimizer:
            self.optimizer.load_state_dict(deepcopy(data["optimizer"]))
        if restore_rng:
            self.rng.bit_generator.state = deepcopy(data["rng"])
            self.torch_rng.set_state(data["torch_rng"].cpu())
        self.updates = int(data["updates"])
        return self

    def save(self, path, **kwargs):
        torch.save(self.state_dict(**kwargs), Path(path))

    def load(self, path, **kwargs):
        return self.load_state_dict(torch.load(Path(path), map_location=self.device, weights_only=True), **kwargs)

    @classmethod
    def from_checkpoint(cls, path, device="cpu", **kwargs):
        data = torch.load(Path(path), map_location=device, weights_only=True)
        learner = cls(data["spec"], {**data["config"], "device": str(device)})
        return learner.load_state_dict(data, **kwargs)


class DQNAgent(_Learner):
    def __init__(self, spec, config=None, **kwargs):
        self._init(spec, _config(DQNConfig, config, kwargs), QNetwork)
        self.online = self.model
        self.target = deepcopy(self.online).eval().requires_grad_(False)
        self.replay = ReplayBuffer(self.spec, self.config.replay_capacity, self.config.seed)

    def act(self, obs, epsilon=0.):
        obs = observation(self.spec, obs)
        epsilon = bounded("epsilon", epsilon)
        if epsilon and self.rng.random() < epsilon:
            return int(self.rng.integers(self.spec.actions))
        with torch.no_grad():
            return int(self.online(obs).argmax(1).item())

    def observe(self, obs, action, reward, next_obs, discount, **flags):
        self.replay.add(obs, action, reward, next_obs, discount, **flags)

    remember = observe

    def update(self, batch_size=None):
        c = self.config
        size = c.batch_size if batch_size is None else positive_int("batch_size", batch_size)
        if len(self.replay) < max(size, c.learning_starts):
            return None
        batch = self.replay.sample(size, self.rng)
        with torch.no_grad():
            targets = double_dqn_targets(batch["rewards"], self.online(batch["next_obs"]),
                                         self.target(batch["next_obs"]), batch["discounts"])
        actions = torch.as_tensor(batch["actions"], device=self.device)
        q = self.online(batch["obs"]).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
        self.optimizer.step()
        self.updates += 1
        synced = self.updates % c.target_every == 0
        if synced:
            self.target.load_state_dict(self.online.state_dict())
        return dict(loss=float(loss.detach()), target_mean=float(targets.mean()), q_mean=float(q.detach().mean()),
                    grad_norm=float(grad), updates=self.updates, target_synced=synced, replay_size=len(self.replay))


class PPOAgent(_Learner):
    """rollout 必须保留采样时 old_logp/values 和 reset 前 next_values。"""
    def __init__(self, spec, config=None, **kwargs):
        self._init(spec, _config(PPOConfig, config, kwargs), ActorCritic)

    def act(self, obs, epsilon=0., *, deterministic=False):
        if bounded("epsilon", epsilon) != 0:
            raise ValueError("PPO 不支持 epsilon 探索；使用策略本身采样")
        obs = observation(self.spec, obs)
        with torch.no_grad():
            logits, value = self.model(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(1) if deterministic else torch.multinomial(
                dist.probs.cpu(), 1, generator=self.torch_rng).squeeze(1).to(self.device)
            return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def value(self, obs):
        with torch.no_grad():
            return float(self.model(observation(self.spec, obs))[1].item())

    def _rollout(self, rollout, gae_lambda):
        required = ("obs", "actions", "rewards", "discounts", "old_logp", "values", "next_values")
        if set(required) - set(rollout):
            raise ValueError(f"rollout 缺少字段 {sorted(set(required)-set(rollout))}")
        raw_obs = rollout["obs"]
        if isinstance(raw_obs, (list, tuple)):
            if not raw_obs:
                raise ValueError("rollout 不能为空")
            raw_obs = _stack([observation(self.spec, x) for x in raw_obs])
        obs = observation(self.spec, raw_obs, batched=True)
        n = len(next(iter(obs.values())))
        data = {key: _array(rollout[key]) for key in required[1:]}
        for key, value in data.items():
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"rollout {key} 必须有限 (T,)")
        actions = data["actions"]
        if np.any((actions < 0) | (actions >= self.spec.actions) | (actions != np.floor(actions))):
            raise ValueError("actions 超出动作空间或不是整数")
        end = np.zeros(n, dtype=bool)
        for key in ("episode_end", "terminated", "truncated"):
            if key in rollout:
                flag = _array(rollout[key])
                if flag.shape != (n,):
                    raise ValueError(f"{key} 必须为 (T,)")
                end |= flag.astype(bool)
                if key == "terminated" and np.any(flag.astype(bool) & (data["discounts"] != 0)):
                    raise ValueError("terminated discount 必须为 0")
        adv, returns = compute_gae(data["rewards"], data["values"], data["next_values"], data["discounts"],
                                   end, gae_lambda=gae_lambda)
        mean, std = float(adv.mean()), float(adv.std())
        if self.config.normalize_advantage and n > 1:
            adv = (adv - mean) / (std + 1e-8)
        tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.device).detach().clone()
        obs = {key: tensor(value, torch.uint8 if key == "image" else torch.float32) for key, value in obs.items()}
        return obs, tensor(actions, torch.long), tensor(data["old_logp"], torch.float32), tensor(adv, torch.float32), tensor(returns, torch.float32), mean, std

    def update(self, rollout, *, gae_lambda=None):
        c = self.config
        obs, actions, old_logp, adv, returns, mean, std = self._rollout(
            rollout, c.gae_lambda if gae_lambda is None else gae_lambda)
        n = len(actions)
        with torch.no_grad():
            logits, _ = self.model(obs)
            initial_ratio = float((torch.distributions.Categorical(logits=logits).log_prob(actions)-old_logp).exp().mean())
        totals = {key: 0. for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "ratio_mean", "grad_norm")}
        samples = minibatches = epochs_completed = 0
        stopped = False
        for epoch in range(c.epochs):
            indices = self.rng.permutation(n)
            for start in range(0, n, c.minibatch_size):
                idx = torch.as_tensor(indices[start:start+c.minibatch_size], device=self.device)
                logits, values = self.model({key: value[idx] for key, value in obs.items()})
                dist = torch.distributions.Categorical(logits=logits)
                log_ratio = dist.log_prob(actions[idx]) - old_logp[idx]
                ratio = log_ratio.exp()
                kl = (ratio - 1 - log_ratio).mean()
                if c.target_kl is not None and float(kl.detach()) > 1.5 * c.target_kl:
                    stopped = True
                    break
                policy = -torch.minimum(ratio*adv[idx], ratio.clamp(1-c.clip_ratio, 1+c.clip_ratio)*adv[idx]).mean()
                value_loss = .5 * F.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()
                loss = policy + c.value_coef*value_loss - c.entropy_coef*entropy
                if not torch.isfinite(loss).item():
                    raise ValueError("PPO loss 非有限")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.optimizer.step()
                metrics = dict(loss=loss, policy_loss=policy, value_loss=value_loss, entropy=entropy,
                               approx_kl=kl, ratio_mean=ratio.mean(), grad_norm=grad)
                for key, value in metrics.items():
                    totals[key] += float(value.detach()) * len(idx)
                samples += len(idx)
                minibatches += 1
            if stopped:
                break
            epochs_completed = epoch+1
        self.updates += 1
        with torch.no_grad():
            logits, _ = self.model(obs)
            log_ratio = torch.distributions.Categorical(logits=logits).log_prob(actions)-old_logp
            final_kl = float((log_ratio.exp()-1-log_ratio).mean())
        return {**{key: value/max(samples, 1) for key, value in totals.items()},
                "initial_ratio_mean": initial_ratio, "final_kl": final_kl, "early_stopped": stopped,
                "epochs_completed": epochs_completed, "minibatches": minibatches, "updates": self.updates,
                "rollout_size": n, "advantage_mean": mean, "advantage_std": std}

