"""新入口生命周期与配置回归；fake 环境不加载全脑或运行正式训练。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from flyrl import runner as r

CONFIG = Path(__file__).resolve().parents[1] / "flyrl/configs/corridor.json"


class FakePipeline:
    dim = 246
    brain = None
    def __init__(self, c):
        self.c = c
        self.calibrated = False
        self.groups = self
        self.seen = 0
    def reset(self):
        self.seen = 0
    def raw(self, rgb):
        self.seen += 1
        return np.full(246, float(rgb[0, 0, 0]) + self.seen / 10, np.float32)
    observe = raw
    def fit(self, data):
        self.calibrated = True
    def save(self, path):
        Path(path).write_bytes(b"fake")


class FakeEnv:
    n_actions = 6
    def __init__(self, *args, **kwargs):
        self.tic = 0
    def reset(self):
        self.tic = 0
        return np.zeros((3, 3, 3), np.uint8), self.info()
    def info(self):
        return {"success": False, "progress": 0., "tics": self.tic, "assets": {}, "layout": None}
    def step(self, action):
        self.tic += 1
        return np.full((3, 3, 3), self.tic, np.uint8), 1., False, self.tic == 2, self.info()
    def close(self):
        pass


class RunnerTests(unittest.TestCase):
    def config(self, **kwargs):
        return r.load_config(CONFIG, kwargs)

    def test_cli_diagnose_dispatch(self):
        from flyrl.cli import main
        with patch("sys.argv", ["flyrl", "diagnose", "--config", str(CONFIG)]), \
             patch("flyrl.diagnostics.diagnose", return_value={"ok": True}) as fn, patch("builtins.print"):
            main()
            self.assertEqual(fn.call_count, 1)

    def test_config_unknown_and_seed_overlap(self):
        for changes in ({"unknown": 3}, {"test_seeds": [41027]}, {"calibration_steps": 1}, {"updates_per_step": 9}):
            with self.assertRaises(ValueError):
                self.config(**changes)

    def test_layout_normalization_and_actual_wad_overlap(self):
        c = self.config()
        self.assertEqual(c["train_layouts"][0]["spawn_jitter"], 8.)
        a = {"layout_id": "entry", "spawn_jitter": 0, "angle_jitter": 0}
        with self.assertRaises(ValueError):
            self.config(train_layouts=[a], val_layouts=[{**a, "layout_seed": 99, "mirror": "left"}])

    def test_calibration_traverses_all_layouts(self):
        c = self.config(calibration_steps=12)
        calls = []
        def make(c, run, split, ep, rng, step=0, calibration=False):
            calls.append((ep, calibration))
            return FakeEnv()
        import time
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, "make_env", make):
            p = FakePipeline(c)
            self.assertEqual(r.calibrate(c, p, Path(tmp), time.perf_counter()+30), 12)
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(flag for _, flag in calls))

    def test_train_dqn_and_ppo_final_snapshot_after_tail(self):
        from flyrl.learning import DQNAgent, PPOAgent
        for algorithm, options in (("dqn", {"batch_size": 2, "learning_starts": 2}),
                                   ("ppo", {"rollout_steps": 128, "epochs": 1, "minibatch_size": 4})):
            with tempfile.TemporaryDirectory() as tmp:
                c = self.config(algorithm=algorithm, output=tmp, max_env_steps=4, checkpoint_every=4,
                                calibration_steps=12, learner=options)
                with patch.object(r, "Pipeline", FakePipeline), patch.object(r, "provenance", return_value={}), \
                     patch.object(r, "make_env", side_effect=lambda *args, **kw: FakeEnv()):
                    result = r.train(c)
                self.assertEqual(result["steps"], 4)
                cls = DQNAgent if algorithm == "dqn" else PPOAgent
                loaded = cls.from_checkpoint(Path(result["checkpoint"]) / "learner.pt")
                self.assertGreater(loaded.updates, 0)
                if algorithm == "ppo":
                    self.assertEqual(loaded.updates, 1)
                self.assertEqual(len(list(Path(result["run"]).glob("checkpoint-*"))), 2)

    def test_anatomical_sides_preserve_midline_and_conflict(self):
        import pandas as pd
        frame = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5], "somaSide": ["L", None, "M", "L", "R"],
                              "rootSide": ["L", "R", "L", "R", None]})
        with patch("pandas.read_feather", return_value=frame):
            mapping, report = r.anatomical_sides("unused", np.array([5, 4, 3, 2, 1]))
        self.assertEqual(mapping, {5:"R", 4:"unknown", 3:"unknown", 2:"R", 1:"L"})
        self.assertEqual(report["conflicts"], [4])

    def test_evaluation_cases_are_unique(self):
        c = self.config()
        cases = []
        with tempfile.TemporaryDirectory() as tmp:
            for episode in range(2):
                env = r.make_env(c, Path(tmp), "val", episode, np.random.default_rng(0))
                cases.append((env.seed, env.layout.layout_seed))
            self.assertEqual(len(set(cases)), 2)
            with self.assertRaises(ValueError):
                r.make_env(c, Path(tmp), "val", 2, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            self.config(calibration_steps=6)

    def test_pipeline_baselines_reset_and_checkpoint(self):
        from flyrl.encoding import Groups
        groups = Groups(np.array([1]), ("one",), (np.array([0]),), np.array([1]), ("",), {})
        uv = np.array([[0., 0.], [1., 1.], [.5, .5]])
        with tempfile.TemporaryDirectory() as tmp:
            graph = Path(tmp) / "graph.npz"
            np.savez(graph, uv=uv, ids=np.array([1]))
            for arm in ("retinal", "pixels"):
                c = self.config(arm=arm, graph=str(graph))
                with patch("flyrl.encoding.build_groups", return_value=groups), patch.object(r, "anatomical_sides", return_value=({}, {})):
                    pipeline = r.Pipeline(c)
                rgb = np.full((30, 82, 3), 90, np.uint8)
                first = pipeline.raw(rgb)
                self.assertEqual(first.shape, (246,))
                pipeline.reset()
                np.testing.assert_array_equal(first, pipeline.raw(rgb))
                pipeline.fit([first, first+1])
                path = Path(tmp) / "pipeline.pt"
                pipeline.save(path)
                pipeline.load(path)
                self.assertTrue(pipeline.calibrated)

    def test_summarize_and_comparison_reject_budget(self):
        rows = [{"success": False, "return": 0., "tics": 2}, {"success": True, "return": 2., "tics": 4}]
        s = r.summarize(rows)
        self.assertEqual(s["success_rate"], .5)
        self.assertLess(s["success_wilson95"][0], .5)
        row = {"config": self.config(), "split": "val", "training_steps": 10, "calibration_frames": 128,
               "sources": {}, "inputs": {}, "versions": {}, "episodes": [], "checkpoint": "x", "summary": s}
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "a.json", Path(tmp) / "b.json"]
            r.json_write(paths[0], row)
            other = copy.deepcopy(row)
            other["config"]["arm"] = "retinal"
            r.json_write(paths[1], other)
            self.assertTrue(r.compare(paths)["comparable"])
            other["config"]["wall_hours"] = 8.
            r.json_write(paths[1], other)
            with self.assertRaises(ValueError):
                r.compare(paths)
            other["config"]["wall_hours"] = 2.
            other["training_steps"] = 9
            r.json_write(paths[1], other)
            with self.assertRaises(ValueError):
                r.compare(paths)


if __name__ == "__main__":
    unittest.main()
