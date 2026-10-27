"""相同小型读出上的无教师在线 DoubleDQN 与 PPO。

只接收编码后的有限小型特征向量，不接收或保存全脑脉冲计数。
CPU 全脑采样昂贵时优先使用 replay 复用数据；device 仅选择读出计算设备，
不会加速脑模拟。两个 actor 均只允许线性或单层 32 单元，critic 可独立选择。
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


__all__ = ["Readout", "ReplayBuffer", "DQNConfig", "DQNAgent", "PPOConfig",
           "PPOAgent", "compute_gae", "double_dqn_targets"]


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return int(value)


def _bounded(name, value, low, high):
    value = float(value)
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} 必须在 [{low}, {high}] 内")
    return value


def _vector(obs, obs_dim):
    value = np.asarray(obs, dtype=np.float32)
    if value.shape != (obs_dim,) or not np.isfinite(value).all():
        raise ValueError(f"obs 必须是有限的 ({obs_dim},) 小型特征向量")
    return value


def _head_sizes(obs_dim, n_actions, hidden):
    _positive_int("obs_dim", obs_dim)
    _positive_int("n_actions", n_actions)
    if hidden not in (0, 32):
        raise ValueError("hidden 只能是 0 或 32，保证两种 actor 容量一致")


class Readout(nn.Module):
    """返回未归一化 Q 或 policy logits；hidden=0 默认线性，32 为单层 Tanh。"""

    def __init__(self, obs_dim, n_actions, hidden=0):
        super().__init__()
        _head_sizes(obs_dim, n_actions, hidden)
        self.obs_dim, self.n_actions, self.hidden = int(obs_dim), int(n_actions), int(hidden)
        self.net = (nn.Linear(self.obs_dim, self.n_actions) if hidden == 0 else
                    nn.Sequential(nn.Linear(self.obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, self.n_actions)))

    def forward(self, obs):
        return self.net(obs)

    def save(self, path):
        """仅保存张量与基本类型，不序列化模型对象。"""
        torch.save({"version": 1, "kind": "Readout", "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions, "hidden": self.hidden,
                    "state_dict": self.state_dict()}, Path(path))

    @classmethod
    def load(cls, path, device="cpu", strict=True):
        data = _read_checkpoint(path, "Readout", device)
        model = cls(data["obs_dim"], data["n_actions"], data["hidden"]).to(device)
        model.load_state_dict(data["state_dict"], strict=strict)
        return model


def _read_checkpoint(path, kind, device):
    data = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(data, dict) or data.get("version") != 1 or data.get("kind") != kind:
        raise ValueError(f"不是受支持的 {kind} checkpoint")
    return data


def double_dqn_targets(rewards, next_online_q, next_target_q, bootstrap_discounts, gamma=.99):
    """在线网络选动作，目标网络估值；discount 为逐转移的无量纲系数。

    target = reward + gamma * bootstrap_discount * Q_target(next_obs,argmax Q_online)。
    terminated 必须对应 discount=0；time-limit truncated 通常为 1，仍用最终观测。
    任一输入是 Tensor 时返回无梯度 Tensor，否则返回 numpy 数组。
    """
    gamma = _bounded("gamma", gamma, 0, 1)
    tensor_mode = any(isinstance(x, torch.Tensor) for x in
                      (rewards, next_online_q, next_target_q, bootstrap_discounts))
    reference = next((x for x in (next_online_q, next_target_q, rewards, bootstrap_discounts)
                      if isinstance(x, torch.Tensor)), None)
    device = reference.device if reference is not None else "cpu"
    with torch.no_grad():
        r, online, target, discount = [torch.as_tensor(x, dtype=torch.float32, device=device)
                                      for x in (rewards, next_online_q, next_target_q,
                                                bootstrap_discounts)]
        if online.ndim != 2 or target.shape != online.shape or online.shape[1] < 1:
            raise ValueError("next Q 必须是同形状的 (N, n_actions) 数组")
        if r.shape != (len(online),) or discount.shape != r.shape:
            raise ValueError("reward 和 bootstrap_discount 必须是 (N,) 数组")
        if not all(torch.isfinite(x).all().item() for x in (r, online, target, discount)):
            raise ValueError("target 输入必须有限")
        if torch.any((discount < 0) | (discount > 1)).item():
            raise ValueError("bootstrap_discount 必须在 [0,1] 内")
        selected = target.gather(1, online.argmax(dim=1, keepdim=True)).squeeze(1)
        result = r + gamma * discount * selected
    return result if tensor_mode else result.numpy()


def compute_gae(rewards, values, next_values, terminated, episode_end=None, *,
                truncated=None, gamma=.99, gae_lambda=.95):
    """返回 (advantages, returns)，支持 (T,) 或时间优先的 (T, n_envs)。

    delta 使用 1-terminated bootstrap；递推使用 1-episode_end。
    truncated 可 bootstrap，但和 terminated 一样强制切断跨 reset 的 GAE。
    next_values 必须由该转移 reset 前的最终观测计算，不能使用下一局初始观测。
    最后一行也允许 bootstrap，但不要求下一段 rollout 的 advantage。
    """
    gamma, lam = _bounded("gamma", gamma, 0, 1), _bounded("gae_lambda", gae_lambda, 0, 1)
    r, v, nv = [np.asarray(x, dtype=np.float64) for x in (rewards, values, next_values)]
    term = np.asarray(terminated, dtype=bool)
    if r.ndim not in (1, 2) or r.shape[0] == 0 or any(x.shape != r.shape for x in (v, nv, term)):
        raise ValueError("GAE 各数组必须同形状，且是非空 (T,) 或 (T,n_envs)")
    if not all(np.isfinite(x).all() for x in (r, v, nv)):
        raise ValueError("GAE 输入必须有限")
    end = term.copy()
    for flag in (episode_end, truncated):
        if flag is not None:
            flag = np.asarray(flag, dtype=bool)
            if flag.shape != r.shape:
                raise ValueError("episode_end/truncated 必须与 reward 同形状")
            end |= flag
    delta = r + gamma * np.where(term, 0., nv) - v
    advantage = np.empty_like(r)
    carry = np.zeros_like(r[0])
    for t in range(len(r) - 1, -1, -1):
        carry = delta[t] + gamma * lam * np.where(end[t], 0., carry)
        advantage[t] = carry
    return advantage.astype(np.float32), (advantage + v).astype(np.float32)


class ReplayBuffer:
    """固定上限环形缓冲区，只保存 float32 特征与显式转移，不保存脑计数。

    bootstrap_discount 默认不是 gamma，而是 terminated=0、其他=1 的系数。
    若环境自动 reset，truncated 转移必须传 final_observation；否则 next_obs
    本身就必须是 reset 前最终观测。本类无法从向量内容判断其来自哪一局。
    """

    def __init__(self, obs_dim, capacity=20_000, seed=None):
        self.obs_dim = _positive_int("obs_dim", obs_dim)
        self.capacity = _positive_int("capacity", capacity)
        if self.capacity > 20_000:
            raise ValueError("replay capacity 不得超过 20000")
        self.obs = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.bootstrap_discounts = np.empty(self.capacity, dtype=np.float32)
        self.size = self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, *, terminated=False, truncated=False,
            bootstrap_discount=None, final_observation=None):
        obs = _vector(obs, self.obs_dim)
        next_obs = _vector(next_obs if final_observation is None else final_observation, self.obs_dim)
        if isinstance(action, (bool, np.bool_)) or int(action) != action or action < 0:
            raise ValueError("action 必须是非负整数")
        if not np.isfinite(reward):
            raise ValueError("reward 必须有限")
        discount = (0. if terminated else 1.) if bootstrap_discount is None else _bounded(
            "bootstrap_discount", bootstrap_discount, 0, 1)
        if terminated and discount != 0:
            raise ValueError("terminated 转移的 bootstrap_discount 必须为 0")
        i = self.position
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i], self.bootstrap_discounts[i] = int(action), reward, discount
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng=None):
        batch_size = _positive_int("batch_size", batch_size)
        if batch_size > self.size:
            raise ValueError("replay 数据不足，不能无放回采样")
        indices = (self.rng if rng is None else rng).choice(self.size, batch_size, replace=False)
        return {name: getattr(self, name)[indices].copy() for name in
                ("obs", "actions", "rewards", "next_obs", "bootstrap_discounts")}

    def state_dict(self):
        """仅保存已填充槽位，避免保存未初始化内存。"""
        return {"obs_dim": self.obs_dim, "capacity": self.capacity, "size": self.size,
                "position": self.position, "rng": deepcopy(self.rng.bit_generator.state),
                **{name: torch.from_numpy(getattr(self, name)[:self.size].copy()) for name in
                   ("obs", "actions", "rewards", "next_obs", "bootstrap_discounts")}}

    def load_state_dict(self, data):
        if data["obs_dim"] != self.obs_dim or data["capacity"] != self.capacity:
            raise ValueError("replay 维度或容量与 checkpoint 不一致")
        size, position = data["size"], data["position"]
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("无效 replay 游标")
        if size < self.capacity and position != size:
            raise ValueError("未填满 replay 的游标必须等于 size")
        for name in ("obs", "actions", "rewards", "next_obs", "bootstrap_discounts"):
            value = data[name].detach().cpu().numpy()
            destination = getattr(self, name)[:size]
            if value.shape != destination.shape or not np.isfinite(value).all():
                raise ValueError(f"无效 replay 字段 {name}")
            destination[:] = value
        if np.any((self.bootstrap_discounts[:size] < 0) | (self.bootstrap_discounts[:size] > 1)):
            raise ValueError("无效 replay bootstrap_discount")
        self.size, self.position = size, position
        self.rng.bit_generator.state = data["rng"]


@dataclass(frozen=True)
class DQNConfig:
    obs_dim: int
    n_actions: int
    hidden: int = 0
    device: str = "cpu"
    seed: int = 0
    lr: float = 3e-4
    gamma: float = .99
    batch_size: int = 64
    replay_capacity: int = 20_000
    learning_starts: int = 256
    target_sync_interval: int = 500
    max_grad_norm: float = 10.

    def __post_init__(self):
        _validate_common(self)
        for name in ("batch_size", "replay_capacity", "learning_starts", "target_sync_interval"):
            _positive_int(name, getattr(self, name))
        if self.replay_capacity > 20_000 or max(self.batch_size, self.learning_starts) > self.replay_capacity:
            raise ValueError("replay 上限为20000，且必须容纳 batch_size 和 learning_starts")


@dataclass(frozen=True)
class PPOConfig:
    obs_dim: int
    n_actions: int
    hidden: int = 0
    critic_hidden: int = 32
    device: str = "cpu"
    seed: int = 0
    lr: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 64
    clip_ratio: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float | None = .03
    normalize_advantage: bool = True

    def __post_init__(self):
        _validate_common(self)
        _head_sizes(self.obs_dim, 1, self.critic_hidden)
        for name in ("rollout_steps", "epochs", "minibatch_size"):
            _positive_int(name, getattr(self, name))
        _bounded("gae_lambda", self.gae_lambda, 0, 1)
        _bounded("clip_ratio", self.clip_ratio, 0, 1)
        for name in ("entropy_coef", "value_coef"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        if self.target_kl is not None and (not np.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl 必须为有限正数或 None")


def _validate_common(config):
    _head_sizes(config.obs_dim, config.n_actions, config.hidden)
    _bounded("gamma", config.gamma, 0, 1)
    if not np.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr 必须是有限正数")
    if not np.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm 必须是有限正数")
    if isinstance(config.seed, bool) or int(config.seed) != config.seed or config.seed < 0:
        raise ValueError("seed 必须是非负整数")


def _config(config_class, config, kwargs):
    if config is None:
        return config_class(**kwargs)
    if isinstance(config, dict):
        return config_class(**{**config, **kwargs})
    if not isinstance(config, config_class):
        raise TypeError(f"config 必须是 {config_class.__name__} 或 dict")
    return config_class(**{**asdict(config), **kwargs})


def _seeded_heads(config, with_critic=False):
    # 仅在局部作用域使用 torch 种子，不改变调用者的全局随机流。
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        actor = Readout(config.obs_dim, config.n_actions, config.hidden)
        critic = Readout(config.obs_dim, 1, config.critic_hidden) if with_critic else None
    return actor.to(config.device), critic.to(config.device) if critic is not None else None


class _CheckpointAgent:
    """所有 checkpoint 都通过 weights_only=True 读取，可严格恢复训练状态。"""

    def _checkpoint(self, include_optimizer, include_rng):
        data = {"version": 1, "kind": type(self).__name__, "config": asdict(self.config),
                "updates": self.updates}
        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()
        if include_rng:
            data["rng"] = deepcopy(self.rng.bit_generator.state)
        return data

    def _restore_common(self, data, strict, restore_optimizer, restore_rng):
        if strict:
            expected = asdict(self.config)
            actual = dict(data["config"])
            # 可在不同设备间恢复；数值运算与随机数序列的逐位一致性不作跨设备保证。
            expected.pop("device")
            actual.pop("device")
            if actual != expected:
                raise ValueError("严格恢复要求除 device 外的 config 完全一致")
        if restore_optimizer:
            if "optimizer" not in data:
                if strict:
                    raise ValueError("checkpoint 未保存 optimizer；请显式关闭恢复")
            else:
                # 优化器恢复也复制张量，后续训练不能回写调用者持有的快照。
                self.optimizer.load_state_dict(deepcopy(data["optimizer"]))
        if restore_rng:
            if "rng" not in data:
                if strict:
                    raise ValueError("checkpoint 未保存 RNG；请显式关闭恢复")
            else:
                self.rng.bit_generator.state = data["rng"]
        self.updates = int(data["updates"])

    @classmethod
    def from_checkpoint(cls, path, device="cpu", *, strict=True,
                        restore_optimizer=True, restore_rng=True, **kwargs):
        data = _read_checkpoint(path, cls.__name__, device)
        agent = cls({**data["config"], "device": str(device)})
        agent.load(path, strict=strict, restore_optimizer=restore_optimizer,
                   restore_rng=restore_rng, **kwargs)
        return agent


class DQNAgent(_CheckpointAgent):
    """在线 DoubleDQN：replay、Huber、梯度裁剪、按优化步数同步 target。

    learn_transition 每新增一个环境转移，最多进行一次优化；未达到 warmup 返回 None。
    update 可额外调用以复用昂贵样本，不会生成环境数据或进行离线 FQI。
    """

    def __init__(self, config=None, **kwargs):
        self.config = _config(DQNConfig, config, kwargs)
        c = self.config
        self.device = torch.device(c.device)
        self.online, _ = _seeded_heads(c)
        self.target = deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=c.lr)
        self.replay = ReplayBuffer(c.obs_dim, c.replay_capacity, c.seed)
        self.rng = np.random.default_rng(c.seed)
        self.updates = 0

    def act(self, obs, epsilon=0., rng=None):
        obs = _vector(obs, self.config.obs_dim)
        epsilon = _bounded("epsilon", epsilon, 0, 1)
        rng = self.rng if rng is None else rng
        if epsilon > 0 and rng.random() < epsilon:
            return int(rng.integers(self.config.n_actions))
        with torch.no_grad():
            return int(self.online(torch.as_tensor(obs, device=self.device)).argmax().item())

    def remember(self, obs, action, reward, next_obs, **flags):
        if isinstance(action, (bool, np.bool_)) or int(action) != action or not 0 <= action < self.config.n_actions:
            raise ValueError("action 超出动作空间")
        self.replay.add(obs, action, reward, next_obs, **flags)

    def learn_transition(self, obs, action, reward, next_obs, **flags):
        self.remember(obs, action, reward, next_obs, **flags)
        return self.update()

    def update(self, batch_size=None, gamma=None):
        c = self.config
        batch_size = c.batch_size if batch_size is None else _positive_int("batch_size", batch_size)
        gamma = c.gamma if gamma is None else _bounded("gamma", gamma, 0, 1)
        if len(self.replay) < max(batch_size, c.learning_starts):
            return None
        batch = {key: torch.as_tensor(value, device=self.device) for key, value in
                 self.replay.sample(batch_size, self.rng).items()}
        with torch.no_grad():
            targets = double_dqn_targets(batch["rewards"], self.online(batch["next_obs"]),
                                         self.target(batch["next_obs"]), batch["bootstrap_discounts"], gamma)
        q = self.online(batch["obs"]).gather(1, batch["actions"][:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), c.max_grad_norm)
        self.optimizer.step()
        self.updates += 1
        synced = self.updates % c.target_sync_interval == 0
        if synced:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.detach()), "q_mean": float(q.detach().mean()),
                "target_mean": float(targets.mean()), "grad_norm": float(grad_norm),
                "updates": self.updates, "target_synced": synced, "replay_size": len(self.replay)}

    def state_dict(self, *, include_optimizer=True, include_rng=True, include_replay=False):
        """返回可独立保存的张量/基本类型快照，不与后续参数更新共享存储。"""
        data = self._checkpoint(include_optimizer, include_rng)
        data.update(online=self.online.state_dict(), target=self.target.state_dict())
        if include_replay:
            data["replay"] = self.replay.state_dict()
        return deepcopy(data)

    def save(self, path, *, include_optimizer=True, include_rng=True, include_replay=False):
        torch.save(self.state_dict(include_optimizer=include_optimizer, include_rng=include_rng,
                                   include_replay=include_replay), Path(path))

    def load(self, path, *, strict=True, restore_optimizer=True, restore_rng=True, restore_replay=False):
        """精确续训需保存并恢复 replay；默认只恢复模型、优化器与动作/采样 RNG。"""
        return self.load_state_dict(_read_checkpoint(path, "DQNAgent", self.device), strict=strict,
                                    restore_optimizer=restore_optimizer, restore_rng=restore_rng,
                                    restore_replay=restore_replay)

    def load_state_dict(self, data, *, strict=True, restore_optimizer=True, restore_rng=True,
                        restore_replay=False):
        if data.get("version") != 1 or data.get("kind") != "DQNAgent":
            raise ValueError("不是受支持的 DQNAgent checkpoint")
        self._restore_common(data, strict, restore_optimizer, restore_rng)
        self.online.load_state_dict(data["online"], strict=strict)
        self.target.load_state_dict(data["target"], strict=strict)
        if restore_replay:
            if "replay" not in data:
                raise ValueError("checkpoint 未保存 replay")
            self.replay.load_state_dict(data["replay"])
            actions = self.replay.actions[:len(self.replay)]
            if np.any((actions < 0) | (actions >= self.config.n_actions)):
                raise ValueError("checkpoint 的 replay action 超出动作空间")
        return self


class PPOAgent(_CheckpointAgent):
    """同一输入/actor 容量的 clipped PPO；rollout 数据必须来自采样时策略。

    rollout_steps 是 trainer 采集建议长度，update 也接受短的最终 rollout。
    rollout 为 dict：obs、actions、rewards、old_logp、values、next_values、
    terminated，另可传 truncated/episode_end。后两者至少应准确标识 reset。
    old_logp 与 values 由 act 返回，next_values 用 value(final_observation) 计算；
    严禁在更新时重新计算 old_logp，否则会破坏重要性比率。
    """

    def __init__(self, config=None, **kwargs):
        self.config = _config(PPOConfig, config, kwargs)
        c = self.config
        self.device = torch.device(c.device)
        self.actor, self.critic = _seeded_heads(c, with_critic=True)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=c.lr)
        self.rng = np.random.default_rng(c.seed)
        # 动作采样使用本地 CPU 随机流，checkpoint 可在 CPU/GPU 之间读取。
        self.torch_rng = torch.Generator(device="cpu").manual_seed(c.seed)
        self.updates = 0

    def act(self, obs, deterministic=False):
        obs = _vector(obs, self.config.obs_dim)
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device)
            distribution = torch.distributions.Categorical(logits=self.actor(x))
            if deterministic:
                action = distribution.logits.argmax()
            else:
                action = torch.multinomial(distribution.probs.cpu(), 1,
                                           generator=self.torch_rng).squeeze(0).to(self.device)
            return int(action.item()), float(distribution.log_prob(action)), float(self.critic(x).squeeze(-1))

    def value(self, obs):
        obs = _vector(obs, self.config.obs_dim)
        with torch.no_grad():
            return float(self.critic(torch.as_tensor(obs, device=self.device)).squeeze(-1))

    def _rollout(self, rollout):
        c = self.config
        required = ("obs", "actions", "rewards", "old_logp", "values", "next_values", "terminated")
        missing = set(required) - set(rollout)
        if missing:
            raise ValueError(f"rollout 缺少字段 {sorted(missing)}")
        # 复制并 detach，保证整个更新过程使用不可变的旧策略概率与旧价值。
        data = {key: (value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor)
                      else np.array(value, copy=True)) for key, value in rollout.items() if key in
                (*required, "truncated", "episode_end")}
        obs = np.asarray(data["obs"], dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != c.obs_dim or len(obs) == 0 or not np.isfinite(obs).all():
            raise ValueError("rollout obs 必须是非空有限 (T,obs_dim) 特征数组")
        n = len(obs)
        for key in required[1:]:
            if data[key].shape != (n,) or not np.isfinite(data[key]).all():
                raise ValueError(f"rollout {key} 必须是有限 (T,) 数组")
        actions = data["actions"]
        if np.any((actions < 0) | (actions >= c.n_actions) | (actions != np.floor(actions))):
            raise ValueError("rollout actions 超出动作空间或不是整数")
        advantages, returns = compute_gae(data["rewards"], data["values"], data["next_values"],
                                         data["terminated"], data.get("episode_end"),
                                         truncated=data.get("truncated"), gamma=c.gamma, gae_lambda=c.gae_lambda)
        adv_mean, adv_std = float(advantages.mean()), float(advantages.std())
        if c.normalize_advantage and n > 1:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=self.device).detach().clone()
        return (tensor(obs, torch.float32), tensor(actions, torch.long),
                tensor(data["old_logp"], torch.float32), tensor(advantages, torch.float32),
                tensor(returns, torch.float32), adv_mean, adv_std)

    def update(self, rollout):
        c = self.config
        obs, actions, old_logp, advantages, returns, adv_mean, adv_std = self._rollout(rollout)
        parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        with torch.no_grad():
            initial_logp = torch.distributions.Categorical(logits=self.actor(obs)).log_prob(actions)
            initial_ratio_mean = float((initial_logp - old_logp).exp().mean())
        totals = {key: 0. for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl",
                                     "clip_fraction", "grad_norm", "ratio_mean")}
        samples, minibatches, epochs_completed, early_stopped = 0, 0, 0, False
        for epoch in range(c.epochs):
            indices = self.rng.permutation(len(obs))
            for start in range(0, len(obs), c.minibatch_size):
                idx = torch.as_tensor(indices[start:start + c.minibatch_size], device=self.device)
                distribution = torch.distributions.Categorical(logits=self.actor(obs[idx]))
                logp = distribution.log_prob(actions[idx])
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                # 更新前检查可避免把已过大偏移的 minibatch 再优化一次。
                if c.target_kl is not None and float(approx_kl.detach()) > 1.5 * c.target_kl:
                    early_stopped = True
                    break
                policy_loss = -torch.minimum(ratio * advantages[idx],
                                             ratio.clamp(1-c.clip_ratio, 1+c.clip_ratio) * advantages[idx]).mean()
                value_loss = .5 * F.mse_loss(self.critic(obs[idx]).squeeze(-1), returns[idx])
                entropy = distribution.entropy().mean()
                loss = policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy
                if not torch.isfinite(loss).item():
                    raise ValueError("PPO loss 非有限；检查 rollout 的旧策略概率和奖励尺度")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(parameters, c.max_grad_norm)
                self.optimizer.step()
                values = {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                          "entropy": entropy, "approx_kl": approx_kl, "grad_norm": grad_norm,
                          "clip_fraction": ((ratio-1).abs() > c.clip_ratio).float().mean(),
                          "ratio_mean": ratio.mean()}
                for key, value in values.items():
                    totals[key] += float(value.detach()) * len(idx)
                samples += len(idx)
                minibatches += 1
            if early_stopped:
                break
            epochs_completed = epoch + 1
        self.updates += 1
        # 最终全 rollout KL 是早停时也可用的监测量，不依赖已执行 minibatch 数。
        with torch.no_grad():
            log_ratio = torch.distributions.Categorical(logits=self.actor(obs)).log_prob(actions) - old_logp
            final_kl = float((log_ratio.exp() - 1 - log_ratio).mean())
        return {**{key: value / max(samples, 1) for key, value in totals.items()},
                "final_kl": final_kl, "initial_ratio_mean": initial_ratio_mean,
                "early_stopped": early_stopped, "epochs_completed": epochs_completed,
                "minibatches": minibatches, "updates": self.updates,
                "advantage_mean": adv_mean, "advantage_std": adv_std,
                "rollout_size": len(obs)}

    def state_dict(self, *, include_optimizer=True, include_rng=True):
        """返回可安全交给 torch.save 的独立快照。"""
        data = self._checkpoint(include_optimizer, include_rng)
        data.update(actor=self.actor.state_dict(), critic=self.critic.state_dict())
        if include_rng:
            data["torch_rng"] = self.torch_rng.get_state()
        return deepcopy(data)

    def save(self, path, *, include_optimizer=True, include_rng=True):
        torch.save(self.state_dict(include_optimizer=include_optimizer, include_rng=include_rng), Path(path))

    def load(self, path, *, strict=True, restore_optimizer=True, restore_rng=True):
        return self.load_state_dict(_read_checkpoint(path, "PPOAgent", self.device), strict=strict,
                                    restore_optimizer=restore_optimizer, restore_rng=restore_rng)

    def load_state_dict(self, data, *, strict=True, restore_optimizer=True, restore_rng=True):
        if data.get("version") != 1 or data.get("kind") != "PPOAgent":
            raise ValueError("不是受支持的 PPOAgent checkpoint")
        self._restore_common(data, strict, restore_optimizer, restore_rng)
        self.actor.load_state_dict(data["actor"], strict=strict)
        self.critic.load_state_dict(data["critic"], strict=strict)
        if restore_rng:
            if "torch_rng" not in data:
                if strict:
                    raise ValueError("checkpoint 未保存 torch RNG")
            else:
                self.torch_rng.set_state(data["torch_rng"].cpu())
        return self
