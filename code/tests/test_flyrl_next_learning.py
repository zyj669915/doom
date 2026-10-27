"""v2 多模态 learner CPU toy 测试；不启动环境、不做长训练。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyrl_next.models import ActorCritic, ModelSpec, QNetwork, observation
from flyrl_next.learning import DQNAgent, PPOAgent, ReplayBuffer, compute_gae, double_dqn_targets


def vector(x=0.):
    return {"vector": np.array([x, 1.], dtype=np.float32)}


def rollout(agent, n=4):
    obs = [vector(i/10) for i in range(n)]
    samples = [agent.act(x) for x in obs]
    return dict(obs=obs, actions=np.array([x[0] for x in samples]),
                old_logp=np.array([x[1] for x in samples], dtype=np.float32),
                values=np.array([x[2] for x in samples], dtype=np.float32),
                next_values=np.array([agent.value(vector((i+1)/10)) for i in range(n)]),
                rewards=np.ones(n), discounts=np.array([.9]*(n-1)+[0.]),
                episode_end=np.array([False]*(n-1)+[True]))


class ModelTests(unittest.TestCase):
    def test_modalities_and_shapes(self):
        for spec, obs in [({"vector_dim": 2, "actions": 3}, vector()),
                          ({"image_shape": (1, 12, 12), "actions": 3}, {"image": np.zeros((1,12,12), np.uint8)}),
                          ({"vector_dim": 2, "image_shape": (1,12,12), "actions": 3},
                           {**vector(), "image": np.full((1,12,12), 255, np.uint8)})]:
            with self.subTest(spec=spec):
                checked = observation(spec, obs)
                self.assertEqual(QNetwork(spec)(checked).shape, (1,3))
                policy, value = ActorCritic(spec)(checked)
                self.assertEqual(policy.shape, (1,3))
                self.assertEqual(value.shape, (1,))
        with self.assertRaises(ValueError):
            ModelSpec()
        with self.assertRaises(ValueError):
            observation({"image_shape": (1,12,12)}, {"image": np.zeros((1,12,12), np.float32)})

    def test_global_rng_not_modified(self):
        before = torch.get_rng_state().clone()
        DQNAgent({"vector_dim": 2})
        PPOAgent({"vector_dim": 2})
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


class ReplayAndMathTests(unittest.TestCase):
    def test_complete_discount_double_dqn(self):
        target = double_dqn_targets([1,2,3], [[5,1],[1,9],[7,0]], [[10,100],[4,8],[99,100]], [.9,.5,0])
        np.testing.assert_allclose(target, [10,6,3])
        online = torch.tensor([[5.,1.]], requires_grad=True)
        out = double_dqn_targets([1], online, [[10,100]], [.9])
        self.assertFalse(out.requires_grad)
        with self.assertRaises(ValueError):
            double_dqn_targets([1], [[1,2]], [[1,2]], [-1])

    def test_dynamic_gae_and_truncation(self):
        adv, ret = compute_gae([1,2,3], [0,0,0], [10,20,30], [.5,.25,0],
                               [False,True,True], gae_lambda=.8)
        np.testing.assert_allclose(adv, [8.8,7,3])
        np.testing.assert_array_equal(adv, ret)
        a, _ = compute_gae(np.ones((2,2)), np.zeros((2,2)), np.zeros((2,2)),
                           np.array([[.5,.25],[0,0]]), gae_lambda=1)
        np.testing.assert_allclose(a, [[1.5,1.25],[1,1]])
        with self.assertRaises(ValueError):
            compute_gae([1],[0],[0],[1.1])

    def test_uint8_ring_and_safe_state(self):
        spec = {"image_shape": (1,8,8), "actions": 2}
        replay = ReplayBuffer(spec, capacity=2)
        obs = {"image": np.full((1,8,8), 7, np.uint8)}
        replay.add(obs, 0, 1., obs, .9)
        obs["image"][:] = 9
        self.assertEqual(replay.storage[0]["obs"]["image"].flat[0], 7)
        replay.add(obs, 1, 2., obs, .8, truncated=True,
                   final_observation={"image": np.full((1,8,8), 3, np.uint8)})
        self.assertEqual(replay.storage[1]["next_obs"]["image"].flat[0], 3)
        replay.add(obs, 0, 3., obs, 0, terminated=True)
        batch = replay.sample(2)
        self.assertEqual(batch["obs"]["image"].dtype, np.uint8)
        self.assertEqual(set(batch["rewards"]), {2,3})
        snap = replay.state_dict()
        restored = ReplayBuffer(spec, 2).load_state_dict(snap)
        self.assertEqual(restored.position, replay.position)
        for a,b in zip(replay.sample(2)["actions"], restored.sample(2)["actions"]):
            self.assertEqual(a,b)
        with self.assertRaises(ValueError):
            replay.add(obs, 0, 0, obs, .9, terminated=True)


class LearnerTests(unittest.TestCase):
    def test_dqn_update_checkpoint_exact_resume(self):
        agent = DQNAgent({"vector_dim": 2}, batch_size=2, replay_capacity=4,
                         learning_starts=2, target_every=1, seed=12)
        self.assertIsNone(agent.update())
        for i in range(4):
            agent.observe(vector(i), i%2, float(i), vector(i+1), .8)
        result = agent.update()
        self.assertTrue(result["target_synced"])
        self.assertTrue(np.isfinite(result["loss"]))
        snap = agent.state_dict()
        self.assertEqual(snap["replay"]["rows"][0]["obs"]["vector"].dtype, torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"dqn.pt"
            agent.save(path)
            torch.load(path, weights_only=True)
            restored = DQNAgent.from_checkpoint(path)
        a, b = agent.update(), restored.update()
        self.assertEqual(a["loss"], b["loss"])
        for key, value in agent.model.state_dict().items():
            torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
        agent.load_state_dict(snap)
        self.assertEqual(agent.updates, 1)
        for key, value in snap["model"].items():
            torch.testing.assert_close(value, agent.model.state_dict()[key], rtol=0, atol=0)
        actions = [agent.act(vector(), epsilon=1) for _ in range(8)]
        self.assertEqual(actions, self._actions_from_snapshot(agent, snap))

    def _actions_from_snapshot(self, agent, snap):
        agent.load_state_dict(snap)
        return [agent.act(vector(),epsilon=1) for _ in range(8)]

    def test_ppo_fixed_old_logp_and_early_stop(self):
        agent = PPOAgent({"vector_dim": 2}, epochs=2, minibatch_size=2, seed=8)
        data = rollout(agent)
        old = data["old_logp"].copy()
        metrics = agent.update(data, gae_lambda=.7)
        self.assertAlmostEqual(metrics["initial_ratio_mean"], 1., places=5)
        self.assertGreater(metrics["minibatches"], 0)
        np.testing.assert_array_equal(data["old_logp"], old)
        stale = rollout(agent)
        stale["old_logp"] += 2.
        metrics = agent.update(stale)
        self.assertAlmostEqual(metrics["initial_ratio_mean"], np.exp(-2), places=5)
        self.assertTrue(metrics["early_stopped"])
        self.assertEqual(metrics["minibatches"], 0)
        self.assertTrue(np.isfinite(metrics["final_kl"]))

    def test_ppo_checkpoint_rng_optimizer(self):
        agent = PPOAgent({"vector_dim": 2}, epochs=1, minibatch_size=2)
        agent.update(rollout(agent))
        snap = agent.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ppo.pt"
            agent.save(path)
            torch.load(path, weights_only=True)
            other = PPOAgent.from_checkpoint(path)
        self.assertEqual([agent.act(vector()) for _ in range(6)], [other.act(vector()) for _ in range(6)])
        agent.load_state_dict(snap)
        other.load_state_dict(snap)
        data = rollout(agent)
        self.assertEqual(agent.update(data)["loss"], other.update(data)["loss"])
        with self.assertRaises(ValueError):
            agent.act(vector(), epsilon=.1)

    def test_image_learner_small_cpu_update(self):
        spec = {"image_shape": (1,8,8), "vector_dim": 2}
        obs = {**vector(), "image": np.zeros((1,8,8), np.uint8)}
        dqn = DQNAgent(spec, batch_size=1, learning_starts=1, replay_capacity=2)
        dqn.observe(obs, dqn.act(obs), 1, obs, .95)
        self.assertTrue(np.isfinite(dqn.update()["loss"]))
        ppo = PPOAgent(spec, epochs=1)
        action, logp, value = ppo.act(obs)
        metrics = ppo.update(dict(obs=[obs], actions=[action], old_logp=[logp], values=[value],
                                  next_values=[ppo.value(obs)], rewards=[1], discounts=[0]))
        self.assertTrue(np.isfinite(metrics["loss"]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
