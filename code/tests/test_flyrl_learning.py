"""小型读出 RL 的确定性 unittest；仅 toy 数据，不启动游戏或脑模拟。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyrl.learning import (DQNAgent, DQNConfig, PPOAgent, PPOConfig, Readout,
                            ReplayBuffer, compute_gae, double_dqn_targets)


class ReadoutTests(unittest.TestCase):
    def test_capacity_and_roundtrip(self):
        for hidden in (0, 32):
            with self.subTest(hidden=hidden), tempfile.TemporaryDirectory() as directory:
                model = Readout(3, 2, hidden)
                expected_count = 3 * 2 + 2 if hidden == 0 else 3 * 32 + 32 + 32 * 2 + 2
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected_count)
                obs = torch.tensor([[1., 2., 3.], [-1., 0., 1.]])
                expected = model(obs).detach().clone()
                path = Path(directory) / "head.pt"
                model.save(path)
                data = torch.load(path, weights_only=True)
                self.assertEqual(data["hidden"], hidden)
                restored = Readout.load(path)
                torch.testing.assert_close(restored(obs), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            Readout(3, 2, hidden=64)

    def test_actor_capacity_is_identical(self):
        for hidden in (0, 32):
            dqn = DQNAgent(obs_dim=3, n_actions=2, hidden=hidden)
            ppo = PPOAgent(obs_dim=3, n_actions=2, hidden=hidden, critic_hidden=32)
            self.assertEqual(sum(p.numel() for p in dqn.online.parameters()),
                             sum(p.numel() for p in ppo.actor.parameters()))

    def test_construction_does_not_change_global_torch_rng(self):
        state = torch.get_rng_state().clone()
        DQNAgent(obs_dim=2, n_actions=2, seed=10)
        PPOAgent(obs_dim=2, n_actions=2, seed=20)
        torch.testing.assert_close(torch.get_rng_state(), state)


class ReplayAndTargetTests(unittest.TestCase):
    def test_double_dqn_selection_uses_online_evaluation_uses_target(self):
        # 第一个 online 选动作0，而 target 的最大值是动作1，不能退化成普通 DQN。
        result = double_dqn_targets([1, 2, 3], [[5, 1], [1, 9], [7, 0]],
                                    [[10, 100], [4, 8], [99, 100]], [1, .5, 0], gamma=.9)
        np.testing.assert_allclose(result, [10., 5.6, 3.], rtol=1e-6)
        online = torch.tensor([[5., 1.]], requires_grad=True)
        target = torch.tensor([[10., 100.]], requires_grad=True)
        value = double_dqn_targets(torch.tensor([1.]), online, target, torch.tensor([1.]), gamma=.9)
        self.assertFalse(value.requires_grad)
        self.assertEqual(value.item(), 10.)
        with self.assertRaises(ValueError):
            double_dqn_targets([1], [[1, 2]], [[1, 2]], [-1])

    def test_terminated_vs_truncated_and_final_observation(self):
        replay = ReplayBuffer(2, capacity=4, seed=0)
        replay.add([1, 2], 0, 1, [3, 4], terminated=True)
        replay.add([5, 6], 1, 2, [-10, -10], truncated=True, final_observation=[7, 8])
        self.assertEqual(replay.bootstrap_discounts[0], 0)
        self.assertEqual(replay.bootstrap_discounts[1], 1)
        np.testing.assert_array_equal(replay.next_obs[1], [7, 8])
        with self.assertRaises(ValueError):
            replay.add([0, 0], 0, 0, [0, 0], terminated=True, bootstrap_discount=1)
        with self.assertRaises(ValueError):
            replay.add([0, 0], 0, float("nan"), [0, 0])
        with self.assertRaises(ValueError):
            replay.add([0, 0, 0], 0, 0, [0, 0])

    def test_bounded_ring_copies_features_and_samples(self):
        replay = ReplayBuffer(2, capacity=3, seed=8)
        obs = np.array([1, 2], dtype=np.float32)
        replay.add(obs, 0, 1, obs)
        obs[:] = 99
        np.testing.assert_array_equal(replay.obs[0], [1, 2])
        for i in range(1, 5):
            replay.add([i, i], 0, i, [i+1, i+1])
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay.position, 2)
        batch = replay.sample(3)
        self.assertEqual(set(batch), {"obs", "actions", "rewards", "next_obs", "bootstrap_discounts"})
        self.assertEqual(set(batch["rewards"]), {2, 3, 4})
        self.assertEqual(batch["obs"].dtype, np.float32)
        batch["obs"][:] = -100
        self.assertTrue((replay.obs >= 2).all())
        with self.assertRaises(ValueError):
            replay.sample(4)
        with self.assertRaises(ValueError):
            ReplayBuffer(2, capacity=20001)

    def test_replay_checkpoint_roundtrip_including_rng_and_cursor(self):
        replay = ReplayBuffer(2, capacity=3, seed=12)
        for i in range(5):
            replay.add([i, i], 0, i, [i+1, i+1])
        restored = ReplayBuffer(2, capacity=3)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.position, replay.position)
        expected, actual = replay.sample(2), restored.sample(2)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        replay.add([9, 9], 1, 9, [10, 10])
        restored.add([9, 9], 1, 9, [10, 10])
        np.testing.assert_array_equal(replay.obs, restored.obs)


class GAETests(unittest.TestCase):
    def test_time_limit_bootstrap_but_no_cross_reset(self):
        args = ([1., 100.], [2., 4.], [10., 0.], [False, True])
        advantage, returns = compute_gae(*args, truncated=[True, False], gamma=.9, gae_lambda=.95)
        np.testing.assert_allclose(advantage, [8., 96.])
        np.testing.assert_allclose(returns, [10., 100.])
        # 没有 reset 标记时，第一行应正常包含第二行的优势。
        advantage, _ = compute_gae(*args, gamma=.9, gae_lambda=.95)
        np.testing.assert_allclose(advantage, [8 + .9*.95*96, 96], rtol=1e-6)
        terminal_adv, _ = compute_gae([1., 100.], [2., 4.], [10., 0.], [True, True], gamma=.9)
        np.testing.assert_allclose(terminal_adv, [-1, 96])

    def test_episode_end_and_terminated_are_always_or_combined(self):
        advantages, _ = compute_gae([1, 100], [0, 0], [10, 0], [True, True],
                                    episode_end=[False, False], gamma=.9)
        np.testing.assert_allclose(advantages, [1, 100])
        advantages, _ = compute_gae([1, 100], [0, 0], [10, 0], [False, True],
                                    episode_end=[True, False], gamma=.9)
        np.testing.assert_allclose(advantages, [10, 100])

    def test_vectorized_shape_and_rollout_tail_bootstrap(self):
        rewards = np.array([[1, 2], [3, 4]], dtype=np.float32)
        values = np.zeros_like(rewards)
        next_values = np.array([[1, 2], [10, 20]], dtype=np.float32)
        terminated = np.array([[False, True], [False, False]])
        result, returns = compute_gae(rewards, values, next_values, terminated, gamma=.5, gae_lambda=1)
        np.testing.assert_allclose(result, [[5.5, 2], [8, 14]])
        np.testing.assert_array_equal(result, returns)
        with self.assertRaises(ValueError):
            compute_gae([1, 2], [0], [0, 0], [False, False])


class DQNTests(unittest.TestCase):
    @staticmethod
    def agent(**overrides):
        return DQNAgent(obs_dim=2, n_actions=2, replay_capacity=64, batch_size=8,
                        learning_starts=8, **overrides)

    @staticmethod
    def fill(agent):
        for i in range(32):
            action = i % 2
            agent.remember([1, 0], action, 1. if action else -1., [0, 0], terminated=True)

    def test_warmup_epsilon_and_target_sync_counts_optimizer_updates(self):
        agent = self.agent(target_sync_interval=2)
        self.assertIsNone(agent.update())
        for _ in range(7):
            self.assertIsNone(agent.learn_transition([1, 0], 0, 1, [1, 0]))
        before = [p.clone() for p in agent.target.parameters()]
        metrics = agent.learn_transition([1, 0], 0, 1, [1, 0])
        self.assertEqual(metrics["updates"], 1)
        self.assertFalse(metrics["target_synced"])
        self.assertTrue(all(torch.equal(p, old) for p, old in zip(agent.target.parameters(), before)))
        metrics = agent.update()
        self.assertEqual(metrics["updates"], 2)
        self.assertTrue(metrics["target_synced"])
        for p, q in zip(agent.online.parameters(), agent.target.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        actions = [agent.act([1, 0], epsilon=1, rng=np.random.default_rng(seed)) for seed in range(30)]
        self.assertEqual(set(actions), {0, 1})
        with self.assertRaises(ValueError):
            agent.remember([1, 0], 2, 0, [1, 0])

    def test_terminal_bandit_loss_decreases(self):
        agent = self.agent(lr=.05)
        self.fill(agent)
        losses = [agent.update()["loss"] for _ in range(100)]
        self.assertLess(np.mean(losses[-10:]), np.mean(losses[:10]) * .02)
        self.assertEqual(agent.act([1, 0]), 1)
        self.assertTrue(all(p.grad is None for p in agent.target.parameters()))

    def test_safe_checkpoint_exact_continuation(self):
        agent = self.agent(target_sync_interval=3)
        self.fill(agent)
        agent.update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dqn.pt"
            agent.save(path, include_replay=True)
            data = torch.load(path, weights_only=True)
            self.assertEqual(data["kind"], "DQNAgent")
            restored = DQNAgent.from_checkpoint(path, restore_replay=True)
            self.assertEqual(restored.updates, 1)
            self.assertEqual([agent.act([1, 0], epsilon=1) for _ in range(20)],
                             [restored.act([1, 0], epsilon=1) for _ in range(20)])
            self.assertEqual(agent.update(), restored.update())
            for p, q in zip(agent.online.parameters(), restored.online.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            bad_config = self.agent(lr=.01)
            with self.assertRaises(ValueError):
                bad_config.load(path)
            agent.save(path, include_optimizer=False, include_rng=False)
            with self.assertRaises(ValueError):
                self.agent(target_sync_interval=3).load(path)
            self.agent(target_sync_interval=3).load(path, restore_optimizer=False, restore_rng=False)
            with self.assertRaises(ValueError):
                self.agent(target_sync_interval=3).load(path, restore_optimizer=False,
                                                        restore_rng=False, restore_replay=True)

    def test_state_snapshot_does_not_alias_live_weights(self):
        agent = self.agent()
        state = agent.state_dict()
        saved = {key: value.clone() for key, value in state["online"].items()}
        self.fill(agent)
        agent.update()
        for key, value in saved.items():
            torch.testing.assert_close(state["online"][key], value, rtol=0, atol=0)


class PPOTests(unittest.TestCase):
    @staticmethod
    def rollout(agent, size=16):
        obs = np.tile([1., 0.], (size, 1)).astype(np.float32)
        outputs = [agent.act(row) for row in obs]
        actions = np.array([row[0] for row in outputs])
        return {"obs": obs, "actions": actions,
                "rewards": np.where(actions == 1, 1., -1.).astype(np.float32),
                "old_logp": np.array([row[1] for row in outputs], dtype=np.float32),
                "values": np.array([row[2] for row in outputs], dtype=np.float32),
                "next_values": np.zeros(size, dtype=np.float32),
                "terminated": np.ones(size, dtype=bool), "truncated": np.zeros(size, dtype=bool),
                "episode_end": np.ones(size, dtype=bool)}

    def test_old_logp_is_fixed_input_not_recomputed(self):
        agent = PPOAgent(obs_dim=2, n_actions=2, lr=1e-5, epochs=3, minibatch_size=8,
                         target_kl=None, normalize_advantage=False)
        rollout = self.rollout(agent)
        # 刻意构造不同的旧概率，若 update 重算 old_logp，这里比率就错误地变成1。
        rollout["old_logp"] -= np.log(2.)
        old_logp = rollout["old_logp"].copy()
        metrics = agent.update(rollout)
        self.assertAlmostEqual(metrics["initial_ratio_mean"], 2., places=5)
        self.assertGreater(metrics["ratio_mean"], 1.99)
        self.assertEqual(metrics["epochs_completed"], 3)
        self.assertEqual(metrics["minibatches"], 6)
        np.testing.assert_array_equal(rollout["old_logp"], old_logp)
        with self.assertRaises(ValueError):
            agent.update({key: value for key, value in rollout.items() if key != "old_logp"})

    def test_tensor_old_probabilities_are_detached(self):
        agent = PPOAgent(obs_dim=2, n_actions=2, epochs=2)
        rollout = self.rollout(agent)
        rollout["old_logp"] = torch.tensor(rollout["old_logp"], requires_grad=True)
        metrics = agent.update(rollout)
        self.assertAlmostEqual(metrics["initial_ratio_mean"], 1., places=5)
        self.assertIsNone(rollout["old_logp"].grad)

    def test_kl_early_stop(self):
        agent = PPOAgent(obs_dim=2, n_actions=2, target_kl=.01)
        rollout = self.rollout(agent)
        rollout["old_logp"] -= np.log(2.)
        state = {key: value.clone() for key, value in agent.actor.state_dict().items()}
        metrics = agent.update(rollout)
        self.assertTrue(metrics["early_stopped"])
        self.assertEqual(metrics["minibatches"], 0)
        self.assertGreater(metrics["final_kl"], .3)
        for key, value in agent.actor.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_toy_bandit_policy_improves_and_value_loss_decreases(self):
        agent = PPOAgent(obs_dim=2, n_actions=2, lr=.02, entropy_coef=0,
                         epochs=4, minibatch_size=32, target_kl=None)
        obs = torch.tensor([1., 0.])
        before = float(agent.actor(obs).softmax(-1)[1].detach())
        for _ in range(20):
            metrics = agent.update(self.rollout(agent, size=64))
            self.assertTrue(np.isfinite(metrics["loss"]))
        after = float(agent.actor(obs).softmax(-1)[1].detach())
        self.assertGreater(after, .95)
        self.assertGreater(after, before + .2)
        # 固定终止奖励仅训练 critic，独立检查可复现的 toy 价值损失下降。
        critic_agent = PPOAgent(obs_dim=2, n_actions=2, lr=.02, epochs=1,
                               entropy_coef=0, target_kl=None, normalize_advantage=False)
        losses = []
        for _ in range(25):
            rollout = self.rollout(critic_agent, size=16)
            rollout["rewards"][:] = .5
            rollout["values"][:] = .5
            losses.append(critic_agent.update(rollout)["value_loss"])
        self.assertLess(np.mean(losses[-5:]), np.mean(losses[:5]) * .2)

    def test_safe_checkpoint_optimizer_and_rng_roundtrip(self):
        agent = PPOAgent(obs_dim=2, n_actions=2, epochs=2)
        agent.update(self.rollout(agent))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"
            agent.save(path)
            self.assertEqual(torch.load(path, weights_only=True)["kind"], "PPOAgent")
            restored = PPOAgent.from_checkpoint(path)
            self.assertEqual([agent.act([1, 0]) for _ in range(20)],
                             [restored.act([1, 0]) for _ in range(20)])
            rollout = self.rollout(agent)
            self.assertEqual(agent.update(rollout), restored.update(rollout))
            for p, q in zip(agent.actor.parameters(), restored.actor.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_defaults_and_invalid_rollout(self):
        config = PPOConfig(obs_dim=2, n_actions=2)
        self.assertEqual((config.rollout_steps, config.epochs, config.minibatch_size), (128, 4, 64))
        self.assertEqual((config.gamma, config.gae_lambda, config.clip_ratio), (.99, .95, .2))
        self.assertEqual(DQNConfig(obs_dim=2, n_actions=2).learning_starts, 256)
        agent = PPOAgent(config)
        rollout = self.rollout(agent)
        rollout["actions"][0] = 8
        with self.assertRaises(ValueError):
            agent.update(rollout)

    @unittest.skipUnless(torch.cuda.is_available(), "本机没有可用 CUDA")
    def test_cuda_head_act_update_and_cross_device_restore(self):
        agent = PPOAgent(obs_dim=2, n_actions=2, device="cuda", epochs=1)
        metrics = agent.update(self.rollout(agent))
        self.assertTrue(np.isfinite(metrics["loss"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.pt"
            agent.save(path)
            cpu = PPOAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(cpu.act([1, 0], deterministic=True)[0],
                             agent.act([1, 0], deterministic=True)[0])
        dqn = DQNAgent(obs_dim=2, n_actions=2, device="cuda", replay_capacity=8,
                       batch_size=2, learning_starts=2)
        dqn.remember([1, 0], 0, 1, [1, 0], truncated=True)
        result = dqn.learn_transition([1, 0], 1, 1, [1, 0], terminated=True)
        self.assertTrue(np.isfinite(result["loss"]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
