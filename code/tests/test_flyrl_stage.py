"""独立协议回归；仅微型模拟数组，不载权威图、不启动ViZDoom。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from flyrl_stage.config import load_config, Curriculum
from flyrl_stage.encoding import Pipeline
from flyrl_stage import runner


class FakeEnv:
    def __init__(self, *args, **kwargs):
        self._done = False
        self.i = 0

    def reset(self):
        return np.full((8, 8, 3), 100, np.uint8), {}

    def step(self, action):
        self.i += 1
        self._done = self.i >= 8
        return np.full((8, 8, 3), 100 + self.i, np.uint8), 1., False, self._done, {
            'reward_terms': {'tic': 1.}, 'success': False, 'wall_hits': 0, 'progress': self.i / 10}

    def close(self):
        pass


class MockMonitor:
    stop_requested = False
    def __init__(self, *args):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


class StageTests(unittest.TestCase):
    def graph(self, directory):
        path = Path(directory) / 'synthetic-test-only.npz'
        np.savez(path, ids=np.array([2**60+1, 2**60+2, 2**60+3], np.int64),
                 retina=np.array([0, 2], np.int32), uv=np.array([[.1, .1], [.9, .9]], np.float32))
        return str(path)

    def test_config_locks_protocol_and_safety(self):
        c = load_config()
        self.assertEqual(c['evaluation_points'], [10000, 30000, 50000])
        self.assertEqual(c['max_game_tics'], 50000)
        for overrides in ({'arm': 'other'}, {'ram_gib': 1.}, {'max_game_tics': 50001},
                          {'curriculum': 'gated'}, {'action_repeat': 1}, {'threshold': -53},
                          {'evaluation_points': [10000, 10000]}, {'intervention': 'black'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config(overrides=overrides)

    def test_duplicate_conditions_and_split_seeds_rejected(self):
        c = load_config()
        for key, value in (('val_layouts', c['train_layouts']), ('val_seeds', c['train_seeds'])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_config(overrides={key: value})

    def test_single_curriculum_can_resume_to_larger_budget(self):
        a = Curriculum(load_config(overrides={'max_game_tics': 32}))
        a.advance(32)
        b = Curriculum(load_config(overrides={'max_game_tics': 64}))
        b.load_state_dict(a.state_dict())
        self.assertEqual(b.game_tics, 32)
        bad = deepcopy(a.state_dict())
        bad['train_layouts'][0]['layout_seed'] += 1
        with self.assertRaises(ValueError):
            b.load_state_dict(bad)

    def test_frontend_hash_precision_reset_calibration_and_safe_state(self):
        with tempfile.TemporaryDirectory() as directory:
            c = load_config(overrides={'arm': 'frontend', 'graph': self.graph(directory)})
            p = Pipeline(c)
            self.assertIsNone(p.brain)
            self.assertEqual(len(p.hash_bucket), 2)
            rgb = np.arange(8*8*3, dtype=np.uint8).reshape(8, 8, 3)
            for image in (rgb, rgb[::-1].copy()):
                p.tic(image)
                p.calibration_update()
            p.freeze_calibration()
            a = p.observation()['vector']
            self.assertEqual(a.shape, (1536,))
            self.assertTrue(np.isfinite(a).all())
            state = p.state_dict()
            path = Path(directory) / 'state.pt'
            torch.save(state, path)
            q = Pipeline(c)
            q.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(q.calibrated)
            np.testing.assert_array_equal(q.mu, p.mu)
            q.reset()
            self.assertEqual(q._tic, 0)
            self.assertFalse(q._hash_rates.any())
            bad = deepcopy(state)
            bad['config']['arm'] = 'real'
            with self.assertRaises(ValueError):
                q.load_state_dict(bad)

    def test_real_and_rewired_preserve_all_small_graph_entries(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_flyrl_next_encoding import EncodingTests
        fixture = EncodingTests()
        fixture.setUp()
        try:
            c = load_config(overrides={'graph': str(fixture.graph)})
            real = Pipeline(c)
            c['arm'] = 'rewired'
            null = Pipeline(c)
            for p in (real, null):
                self.assertEqual(p.brain.n, 4)
                self.assertEqual(len(p.brain.post), 5)
                p.tic(fixture.white)
                self.assertTrue(np.isfinite(p.observation()['vector']).all())
            np.testing.assert_array_equal(real.hash_bucket, null.hash_bucket)
            np.testing.assert_array_equal(real.brain.weight, null.brain.weight)
            self.assertIsNotNone(null.brain.rewire_report)
            self.assertNotEqual(real.graph_signature, null.graph_signature)
        finally:
            fixture.doCleanups()

    def test_macro_partial_truncation_and_reward(self):
        class P:
            calls = 0
            last_diagnostic = {'actual_ms': 28.5}
            def tic(self, rgb):
                self.calls += 1
            def observation(self):
                return {'vector': np.array([self.calls], np.float32)}
        c = load_config()
        c['gamma_tic'] = .9
        p = P()
        result = runner.macro_step(FakeEnv(), p, 0, c, remaining=2)
        self.assertAlmostEqual(result[1], 1.9)
        self.assertAlmostEqual(result[2], .81)
        self.assertEqual(p.calls, 2)
        self.assertEqual(result[-1]['macro_tics'], 2)

    def test_gate_rejection_never_constructs_brain_or_agent(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'gate', return_value={
                'allowed': False, 'errors': ['mock insufficient']}), patch.object(runner, 'Pipeline') as build:
            result = runner.train(load_config(overrides={'output': directory}))
            self.assertFalse(result['started'])
            self.assertEqual(result['stop'], 'resource_gate_rejected')
            build.assert_not_called()

    def test_real_incremental_resume_and_prelocked_points_without_real_game(self):
        with tempfile.TemporaryDirectory() as directory:
            c = load_config(overrides={'arm': 'frontend', 'output': directory,
                'graph': self.graph(directory), 'max_game_tics': 32, 'calibration_steps': 24,
                'evaluation_points': [16, 32, 64], 'checkpoint_every': 4,
                'learner': {'replay_capacity': 64, 'batch_size': 2, 'learning_starts': 2, 'target_every': 2}})
            with patch.object(runner, 'gate', return_value={'allowed': True, 'errors': []}), \
                 patch.object(runner, 'MemoryMonitor', MockMonitor), \
                 patch.object(runner, 'provenance', return_value={'synthetic': True}), \
                 patch.object(runner, 'make_env', side_effect=lambda *args: FakeEnv()):
                first = runner.train(c)
                self.assertEqual(first['game_tics'], 32)
                self.assertGreater(first['updates'], 0)
                rows = [json.loads(x) for x in (Path(first['run']) / 'evaluation-queue.jsonl').read_text().splitlines()]
                self.assertEqual([x['game_tics'] for x in rows], [0, 16, 32])
                c['max_game_tics'] = 64
                second = runner.train(c, resume=first['checkpoint'])
                self.assertEqual(second['game_tics'], 64)
                self.assertGreater(second['updates'], first['updates'])
                self.assertGreater(second['replay_size'], first['replay_size'])
                evaluated = runner.evaluate(second['checkpoint'], quick=True)
                self.assertTrue(evaluated['complete'])
                self.assertEqual(len(evaluated['episodes']), 4)
                self.assertEqual(evaluated['training_game_tics'], 64)
                manifest = Path(second['checkpoint']) / 'state.pt'
                with open(manifest, 'ab') as stream:
                    stream.write(b'tampered')
                with self.assertRaises(ValueError):
                    runner.load_checkpoint(second['checkpoint'])

    def test_controlled_stop_and_atomic_failure(self):
        c = load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'STOP').touch()
            self.assertEqual(runner.stop_reason(c, path, MockMonitor(), 0), 'controlled_stop')
            class S:
                def state_dict(self):
                    return {}
            with patch.object(runner.torch, 'save', side_effect=OSError('mock disk failure')):
                with self.assertRaises(OSError):
                    runner.checkpoint(path, {}, {}, S(), S(), {'game_tics': 1})
            self.assertFalse((path / 'latest-checkpoint.json').exists())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
