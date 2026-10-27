"""真实 runner/learner 与假逐 tic 环境的短程续训集成测试。"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyrl_next import runner
from flyrl_next.config import load_config


class FakePipeline:
    instances = []

    def __init__(self, config, path):
        self.spec = {"vector_dim": 2}
        self.calibrated = False
        self.calibration_count = 0
        self.sd = np.ones(2, np.float32)
        self.frame = np.zeros(2, np.float32)
        self.last_diagnostic = {"actual_ms": 0.}
        self.resets = 0
        self.__class__.instances.append(self)

    def reset(self):
        self.resets += 1
        self.frame[:] = -999

    def tic(self, rgb):
        self.frame = np.array(rgb, dtype=np.float32, copy=True)

    def observation(self):
        return {"vector": self.frame.copy()}

    def calibration_update(self):
        self.calibration_count += 1

    def freeze_calibration(self):
        self.calibrated = True

    def state_dict(self):
        return {"calibrated": self.calibrated}

    def load_state_dict(self, data):
        self.calibrated = data["calibrated"]


class FakeEnv:
    instances = []
    training_steps = 0
    stop_at = None
    stop_path = None

    def __init__(self, config, path, layout, seed):
        self.number = len(self.__class__.instances) + 1
        self.__class__.instances.append(self)
        self._done = False
        self._telemetry = 0
        self.tics = 0
        self.closed = False
        self.resets = 0

    def _info(self, telemetry):
        return {"progress": float(telemetry), "reward_terms": {"tic": 1.},
                "success": False, "wall_hits": 0, "telemetry": {"tics": self.tics}}

    def reset(self):
        self.resets += 1
        self.tics = self._telemetry = 0
        self._done = False
        return np.array([self.number, 0], np.float32), self._info(0)

    def step(self, action):
        self.tics += 1
        self._telemetry = self.tics
        if FakePipeline.instances[-1].calibrated:
            self.__class__.training_steps += 1
            if self.__class__.training_steps == self.stop_at:
                self.stop_path.touch()
        return np.array([self.number, self.tics], np.float32), 1., False, False, self._info(self.tics)

    def close(self):
        self.closed = True
        self._done = True


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        FakePipeline.instances = []
        FakeEnv.instances = []
        FakeEnv.training_steps = 0
        FakeEnv.stop_at = 3
        FakeEnv.stop_path = self.root / "STOP"
        self.agents = []
        self.epsilons = []
        real_make_agent = runner.make_agent

        def make_agent(c, pipeline):
            agent = real_make_agent(c, pipeline)
            self.agents.append(agent)
            if c["algorithm"] == "dqn":
                original = agent.act

                def act(obs, epsilon=0):
                    self.epsilons.append(epsilon)
                    return original(obs, epsilon=epsilon)
                agent.act = act
            return agent

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, "Pipeline", FakePipeline))
        self.stack.enter_context(patch.object(runner, "make_env", FakeEnv))
        self.stack.enter_context(patch.object(runner, "make_agent", make_agent))
        self.provenance = self.stack.enter_context(patch.object(runner, "provenance", return_value={"source": "fixed-test-source"}))
        self.stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))

    def config(self, route="A6", **kwargs):
        learner = ({"batch_size": 1, "learning_starts": 1, "replay_capacity": 10, "target_every": 2}
                   if route == "A6" else {"epochs": 1, "minibatch_size": 2})
        return load_config(overrides={"route": route, "output": str(self.root), "action_repeat": 1,
                                      "calibration_steps": 24, "checkpoint_every": 2, "max_game_tics": 6,
                                      "epsilon_decay_steps": 10, "epsilon_end": .1, "rollout_steps": 3,
                                      "learner": learner, **kwargs})

    def test_dqn_stop_resume_preserves_schedule_replay_optimizer_new_episode(self):
        c = self.config()
        first = runner.train(c)
        self.assertEqual(first["stop"], "controlled_stop")
        self.assertEqual(first["game_tics"], 3)
        self.assertEqual(first["decisions"], 3)
        saved = torch.load(Path(first["checkpoint"])/"state.pt", map_location="cpu", weights_only=True)
        learner = saved["learner"]
        self.assertEqual(len(learner["replay"]["rows"]), 3)
        self.assertEqual(learner["updates"], 3)
        self.assertTrue(learner["optimizer"]["state"])
        torch.testing.assert_close(learner["replay"]["rows"][-1]["next_obs"]["vector"],
                                   torch.tensor([float(FakeEnv.instances[-1].number), 3.]))
        self.assertTrue(all(env.closed for env in FakeEnv.instances))
        np.testing.assert_allclose(self.epsilons, [1., .91, .82], rtol=0, atol=1e-12)
        FakeEnv.stop_path.unlink()
        before_envs = len(FakeEnv.instances)
        before_calibration = first["calibration_tics"]
        resumed = runner.train(c, resume=first["checkpoint"])
        self.assertEqual(resumed["game_tics"], 6)
        self.assertEqual(resumed["decisions"], 6)
        self.assertEqual(resumed["calibration_tics"], before_calibration)
        self.assertEqual(len(FakeEnv.instances), before_envs + 1)
        self.assertEqual(FakeEnv.instances[-1].resets, 1)
        self.assertEqual(FakePipeline.instances[-1].resets, 1)
        self.assertAlmostEqual(self.epsilons[3], .73)
        final = torch.load(Path(resumed["checkpoint"])/"state.pt", weights_only=True)["learner"]
        self.assertEqual(len(final["replay"]["rows"]), 6)
        self.assertEqual(final["updates"], 6)
        for key, value in learner["optimizer"]["state"].items():
            self.assertEqual(float(final["optimizer"]["state"][key]["step"]), float(value["step"]) + 3)
        for old, new in zip(learner["replay"]["rows"], final["replay"]["rows"][:3]):
            torch.testing.assert_close(old["obs"]["vector"], new["obs"]["vector"], rtol=0, atol=0)
        first_new = final["replay"]["rows"][3]["obs"]["vector"]
        torch.testing.assert_close(first_new, torch.tensor([float(FakeEnv.instances[-1].number), 0.]))
        self.assertEqual(resumed["resume_semantics"], "new_episode")

    def test_ppo_checkpoint_tail_is_discarded_and_accounted_on_resume(self):
        c = self.config("A5")
        first = runner.train(c)
        self.assertEqual(first["ppo_discarded_tail"], 3)
        data = torch.load(Path(first["checkpoint"])/"state.pt", weights_only=True)
        self.assertEqual(data["learner"]["updates"], 0)
        self.assertNotIn("rollout", data)
        FakeEnv.stop_path.unlink()
        resumed = runner.train(c, resume=first["checkpoint"])
        self.assertEqual(resumed["ppo_discarded_tail"], 6)
        final = torch.load(Path(resumed["checkpoint"])/"state.pt", weights_only=True)
        self.assertEqual(final["learner"]["updates"], 0)
        self.assertEqual(final["state"]["decisions"], 6)

    def test_ppo_full_rollout_updates_and_discards_only_tail(self):
        FakeEnv.stop_at = None
        result = runner.train(self.config("A5", checkpoint_every=100, max_game_tics=5))
        data = torch.load(Path(result["checkpoint"])/"state.pt", weights_only=True)
        self.assertEqual(data["learner"]["updates"], 1)
        self.assertTrue(data["learner"]["optimizer"]["state"])
        self.assertEqual(result["ppo_discarded_tail"], 2)
        rows = [json.loads(x) for x in (Path(result["run"])/"updates.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rollout_size"], 3)
        self.assertAlmostEqual(rows[0]["initial_ratio_mean"], 1., places=5)

    def test_hash_tamper_and_source_drift_are_rejected(self):
        result = runner.train(self.config())
        path = Path(result["checkpoint"])
        runner.load_checkpoint(path)
        self.provenance.return_value = {"source": "changed"}
        with self.assertRaisesRegex(ValueError, "漂移"):
            runner.load_checkpoint(path)
        self.provenance.return_value = {"source": "fixed-test-source"}
        with (path/"state.pt").open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "哈希"):
            runner.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
