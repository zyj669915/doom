"""短任务适配纯回归：替身环境/小向量，使用真实DoubleDQN，不读权威脑图。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flyrl_turn_train import config, runner
from flyrl_turn_train.encoding import Pipeline
from flyrl_stage.runner import macro_step
from flyrl_turn.config import manifest


class Monitor:
    stop_requested = False


class TinyPipeline:
    def __init__(self, c, path=None):
        self.spec = {'vector_dim': 4, 'image_shape': None}
        self.mu = np.zeros(4, np.float32)
        self.sd = np.ones(4, np.float32)
        self.calibrated = False
        self.calibration_count = 0
        self.mode = 'none'
        self.resets = 0
        self.reset()

    def reset(self):
        self.resets += 1
        self.value = 0.
        self.last_diagnostic = {'actual_ms': 0.}

    def tic(self, rgb):
        self.value = 0. if self.mode == 'black' else float(rgb[0, 0, 0])
        self.last_diagnostic = {'actual_ms': 1.}

    def observation(self):
        return {'vector': np.full(4, self.value, np.float32)}

    def calibration_update(self):
        if self.calibrated:
            raise RuntimeError('禁止冻结后校准')
        self.calibration_count += 1

    def freeze_calibration(self):
        self.calibrated = True

    def state_dict(self):
        return {'calibrated': self.calibrated, 'count': self.calibration_count}

    def load_state_dict(self, value):
        self.calibrated, self.calibration_count = value['calibrated'], value['count']
        self.reset()

    def set_intervention(self, mode):
        self.mode = mode
        self.reset()


class TinyEnv:
    made = []
    stop_at_close = None

    def __init__(self, c, path, case):
        self.case = deepcopy(case)
        self.path = path
        self._done = True
        self.tics = 0
        self.closed = False
        TinyEnv.made.append(self)

    def reset(self):
        self._done = False
        self.tics = 0
        return np.ones((2, 2, 3), np.uint8), {'secret': '不得进入agent', 'assets': {'test_only': True}}

    def step(self, action):
        self.tics += 1
        self._done = self.tics >= 5
        rgb = np.full((2, 2, 3), self.tics + 1, np.uint8)
        info = {'reward_terms': {'tic': 1.}, 'progress': self.tics / 5,
                'success': self._done, 'wall_hits': 0, 'reason': 'short_goal' if self._done else None}
        return rgb, 1., self._done, False, info

    def close(self):
        self.closed = True
        if TinyEnv.stop_at_close:
            TinyEnv.stop_at_close(self)


def small_config(**values):
    return config.load_config(values={'max_game_tics': 12, 'calibration_steps': 24,
        'learner': {'batch_size': 2, 'learning_starts': 2, 'replay_capacity': 8, 'target_every': 2}, **values})


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'run'
        self.path.mkdir()
        TinyEnv.made = []
        TinyEnv.stop_at_close = None
        self.saved = []
        self.patches = [patch.object(runner, 'Pipeline', TinyPipeline),
                        patch.object(runner, 'make_env', TinyEnv),
                        patch.object(runner, 'provenance', return_value={'test': True}),
                        patch.object(runner, 'save_checkpoint', side_effect=self.save)]
        for p in self.patches:
            p.start()
        self.addCleanup(self.tmp.cleanup)
        for p in self.patches:
            self.addCleanup(p.stop)

    def save(self, path, c, prov, pipeline, agent, state):
        data = deepcopy({'pipeline': pipeline.state_dict(), 'learner': agent.state_dict(),
                         'state': state, 'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state()})
        self.saved.append(data)
        return path / f'checkpoint-test-{len(self.saved)}'

    def train(self, c=None, **kwargs):
        return runner.run_train(c or small_config(), self.path, Monitor(), **kwargs)

    def test_complete_training_calibration_separate_and_no_telemetry(self):
        result = self.train()
        self.assertTrue(result['complete'])
        self.assertEqual(12, result['game_tics'])
        self.assertGreater(result['calibration_tics'], 0)
        self.assertEqual(24, self.saved[-1]['pipeline']['count'])
        self.assertGreater(result['updates'], 0)
        self.assertEqual(result['decisions'], result['replay_size'])
        self.assertTrue(all(e.closed for e in TinyEnv.made))
        train_cases = {f'train-{i:02d}' for i in range(12)}
        self.assertTrue(all(e.case['case_id'] in train_cases for e in TinyEnv.made))
        self.assertEqual({'vector'}, set(self.saved[-1]['learner']['replay']['rows'][0]['obs']))
        self.assertEqual(0, self.saved[0]['state']['game_tics'])
        self.assertEqual(0, self.saved[0]['learner']['updates'])
        self.assertTrue((self.path / 'assets.jsonl').exists())
        rows = [json.loads(x) for x in (self.path / 'assets.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(r['assets'] == {'test_only': True} for r in rows))

    def test_resume_preserves_optimizer_replay_rng_and_skips_calibration(self):
        first = self.train()
        data = self.saved[-1]
        initial = len(TinyEnv.made)
        c = small_config(max_game_tics=20)
        with patch.object(runner, 'load_checkpoint', return_value=(c, {}, data)):
            result = self.train(c, resume='fake')
        self.assertEqual(20, result['game_tics'])
        self.assertEqual(first['calibration_tics'], result['calibration_tics'])
        self.assertGreater(result['updates'], first['updates'])
        self.assertTrue(self.saved[-1]['learner']['optimizer']['state'])
        self.assertGreater(len(TinyEnv.made), initial)
        self.assertEqual('new_episode; brain/game/history reset', result['resume_semantics'])
        self.assertEqual(first['wall_s'], result['previous_wall_s'])

    def test_reject_uncalibrated_or_smoke_resume(self):
        self.train()
        for key, value in (('smoke_only', True), ('protocol', 'old-protocol')):
            data = deepcopy(self.saved[-1])
            data['state'][key] = value
            with patch.object(runner, 'load_checkpoint', return_value=(small_config(), {}, data)):
                with self.assertRaises(ValueError):
                    self.train(resume='bad')
        data = deepcopy(self.saved[-1])
        data['pipeline']['calibrated'] = False
        with patch.object(runner, 'load_checkpoint', return_value=(small_config(), {}, data)):
            with self.assertRaises(ValueError):
                self.train(resume='uncalibrated')

    def test_accept_does_not_calibrate_optimize_or_store_replay(self):
        result = self.train(small_config(max_game_tics=300), smoke=True)
        self.assertTrue(result['passed'])
        self.assertEqual(256, result['game_tics'] + result['initial_observation_tics'])
        self.assertEqual(0, result['updates'])
        self.assertEqual(0, result['replay_size'])
        self.assertEqual(0, result['calibration_tics'])
        self.assertFalse(result['training_started'])
        self.assertFalse((self.path / 'evaluation-queue.jsonl').exists())

    def test_stop_before_pipeline_does_not_build_anything(self):
        (self.path.parent / 'STOP').touch()
        result = self.train()
        self.assertFalse(result['started'])
        self.assertEqual('controlled_stop', result['stop'])
        self.assertEqual([], TinyEnv.made)

    def test_last_close_stop_vetoes_train_completion(self):
        c = small_config(calibration_steps=24)
        def on_close(env):
            if len(TinyEnv.made) > 12:
                (self.path / 'STOP').touch()
        TinyEnv.stop_at_close = on_close
        result = self.train(c)
        self.assertFalse(result['complete'])
        self.assertEqual('controlled_stop', result['stop'])
        self.assertTrue(self.saved)

    def test_eval_frozen_dev_only_and_black_no_calibration(self):
        self.train()
        data = deepcopy(self.saved[-1])
        prior_saved = len(self.saved)
        for policy in ('learned', 'untrained', 'random', 'sequence-negative', 'sequence-positive'):
            TinyEnv.made = []
            with patch.object(runner, 'load_checkpoint', return_value=(small_config(), {'provenance': {'test': True}, 'state_sha256': 'x'}, data)):
                result = runner.run_evaluate(small_config(), self.path, Monitor(), 'fake',
                                             quick=True, policy=policy, intervention='black')
            self.assertTrue(result['complete'])
            self.assertEqual(4, len(result['episodes']))
            self.assertEqual(['dev-00', 'dev-01', 'dev-02', 'dev-03'], [e.case['case_id'] for e in TinyEnv.made])
            self.assertEqual(prior_saved, len(self.saved))
        self.assertEqual(data['learner']['updates'], self.saved[-1]['learner']['updates'])

    def test_eval_final_close_stop_keeps_rows_but_null_rate(self):
        self.train()
        data = self.saved[-1]
        TinyEnv.made = []
        TinyEnv.stop_at_close = lambda env: (self.path / 'STOP').touch() if len(TinyEnv.made) == 4 else None
        with patch.object(runner, 'load_checkpoint', return_value=(small_config(), {'provenance': {'test': True}, 'state_sha256': 'x'}, data)):
            result = runner.run_evaluate(small_config(), self.path, Monitor(), 'fake', quick=True)
        self.assertEqual(4, len(result['episodes']))
        self.assertFalse(result['complete'])
        self.assertIsNone(result['success_rate'])

    def test_smaller_resume_budget_is_rejected_before_pipeline(self):
        self.train()
        data = self.saved[-1]
        with patch.object(runner, 'load_checkpoint', return_value=(small_config(), {}, data)):
            with self.assertRaises(ValueError):
                self.train(small_config(max_game_tics=4), resume='fake')

    def test_stop_during_calibration_saves_nonresumable_partial(self):
        TinyEnv.stop_at_close = lambda env: (self.path / 'STOP').touch()
        result = self.train()
        self.assertEqual('calibration', result['phase'])
        self.assertEqual(0, result['game_tics'])
        self.assertFalse(result['complete'])
        self.assertFalse(self.saved[-1]['pipeline']['calibrated'])
        self.assertEqual(0, result['updates'])

    def test_true_checkpoint_roundtrip_with_tiny_environment(self):
        from flyrl_turn_train import storage
        from test_flyrl_turn_train_storage import fake_provenance
        prov = fake_provenance()
        with patch.object(storage, 'provenance', return_value=prov), \
             patch.object(runner, 'provenance', return_value=prov), \
             patch.object(runner, 'save_checkpoint', wraps=storage.save_checkpoint), \
             patch.object(runner, 'load_checkpoint', wraps=storage.load_checkpoint):
            first = self.train()
            checkpoint = Path(first['checkpoint'])
            self.assertTrue((checkpoint / 'manifest.json').is_file())
            second_path = self.path / 'resumed'
            second_path.mkdir()
            second = runner.run_train(small_config(max_game_tics=20), second_path, Monitor(), resume=checkpoint)
            self.assertTrue(second['complete'])
            self.assertEqual(20, second['game_tics'])
            self.assertEqual(first['calibration_tics'], second['calibration_tics'])
            _, _, state = storage.load_checkpoint(second['checkpoint'])
            self.assertEqual(second['updates'], state['learner']['updates'])
            self.assertEqual(8, len(state['learner']['replay']['rows']))
            self.assertTrue(state['learner']['optimizer']['state'])

    def test_exception_cost_logged_without_saving_partial_learner(self):
        class FailingEnv(TinyEnv):
            def step(self, action):
                self.tics += 1
                raise FloatingPointError('fake nonfinite after game advanced')
        with patch.object(runner, 'make_env', FailingEnv):
            with self.assertRaises(FloatingPointError):
                self.train()
        progress = json.loads((self.path / 'failure-progress.json').read_text(encoding='utf-8'))
        self.assertEqual(1, progress['unreturned_macro_game_tics'])
        self.assertFalse(progress['automatic_resume_allowed'])
        self.assertEqual([], self.saved)

    def test_failed_run_checkpoint_resume_rejected(self):
        checkpoint = self.path / 'checkpoint-safe'
        checkpoint.mkdir()
        (self.path / 'failure.json').write_text('{}', encoding='utf-8')
        with patch.object(runner, 'load_checkpoint') as loader:
            with self.assertRaises(ValueError):
                self.train(resume=checkpoint)
        loader.assert_not_called()

    def test_source_drift_vetoes_complete_eval(self):
        self.train()
        data = self.saved[-1]
        with patch.object(runner, 'load_checkpoint', return_value=(small_config(), {'provenance': {'test': True}, 'state_sha256': 'x'}, data)), \
             patch.object(runner, 'provenance', return_value={'changed': True}):
            result = runner.run_evaluate(small_config(), self.path, Monitor(), 'fake', quick=True)
        self.assertEqual(4, len(result['episodes']))
        self.assertEqual('source_drift', result['stop'])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['success_rate'])

    def test_macro_peak_records_intermediate_progress(self):
        class PeakEnv(TinyEnv):
            def step(self, action):
                rgb, r, term, trunc, info = super().step(action)
                info['progress'] = [.40, .43, .42, .41][self.tics - 1]
                return rgb, r, term, trunc, info
        c = small_config()
        env = PeakEnv(c, self.path, {})
        env.reset()
        _, reward, _, _, _, info = runner.macro_step(env, TinyPipeline(c), 0, c)
        self.assertEqual(.43, info['macro_max_progress'])
        row = runner.episode_row({'case_id': 'x', 'seed': 1, 'layout': {}}, 0)
        runner.record_step(row, 0, reward, info, c)
        self.assertEqual(.43, row['max_progress'])

    def test_wall_and_resource_stops(self):
        self.assertEqual('wall_limit', runner.stop_reason(self.path, Monitor(), time.monotonic()-100, 1))
        m = Monitor()
        m.stop_requested = True
        self.assertEqual('resource_stop', runner.stop_reason(self.path, m, time.monotonic(), 100))


class PureContractTests(unittest.TestCase):
    def test_config_strict_and_snapshot_detached(self):
        raw = {'learner': {'lr': .001}}
        c = config.load_config(values=raw)
        raw['learner']['lr'] = 9
        self.assertEqual(.001, c['learner']['lr'])
        for bad in ({'protocol': 'malecns-near-turn-three-arm-v1'}, {'arm': 'rewired'},
                    {'max_tics': 1200}, {'gamma_tic': True}, {'ram_gib': 1.},
                    {'commit_gib': float('nan')}, {'max_game_tics': 30001},
                    {'learner': {'device': 'cuda'}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                config.load_config(values=bad)

    def test_conditions_unique_and_no_old_test(self):
        m = manifest()
        self.assertEqual({'train', 'dev', 'confirm'}, set(m['splits']))
        hashes = [r['wad_sha256'] for rows in m['splits'].values() for r in rows]
        self.assertEqual(48, len(set(hashes)))

    def test_macro_partial_terminal_discount_and_per_tic_encoding(self):
        c = small_config()
        p = TinyPipeline(c)
        env = TinyEnv(c, Path('.'), {'case_id': 'x'})
        env.reset()
        obs, reward, discount, term, trunc, info = macro_step(env, p, 3, c, 2)
        self.assertAlmostEqual(1 + .999, reward)
        self.assertAlmostEqual(.999**2, discount)
        self.assertEqual(2, info['macro_tics'])
        self.assertEqual(3, obs['vector'][0])
        _, _, discount, term, _, info = macro_step(env, p, 3, c)
        self.assertTrue(term)
        self.assertEqual(0., discount)
        self.assertEqual(3, info['macro_tics'])

    def test_truncation_uses_true_final_rgb_and_bootstraps(self):
        c = small_config()
        p = TinyPipeline(c)
        class TruncEnv:
            def step(self, action):
                return np.full((2, 2, 3), 7, np.uint8), 1., False, True, {'reward_terms': {}}
        obs, _, discount, term, trunc, info = macro_step(TruncEnv(), p, 0, c)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(.999, discount)
        self.assertEqual(7., obs['vector'][0])
        self.assertEqual(1., info['brain_ms'])

    def test_frontend_no_brain_import_and_encoding_protocol_bound(self):
        with tempfile.TemporaryDirectory() as name:
            graph = Path(name) / 'tiny.npz'
            np.savez(graph, uv=np.array([[.25, .5], [.75, .5]]), ids=np.array([1, 2]), retina=np.array([0, 1]))
            c = small_config(arm='frontend', graph=str(graph))
            # 缺少ptr/post/weight等脑必需字段；此路径若实例化脑必定失败。
            p = Pipeline(c)
            self.assertIsNone(p.brain)
            self.assertEqual(config.PROTOCOL, p.encoding_config['protocol'])
            rgb = np.full((2, 2, 3), 255, np.uint8)
            p.tic(rgb)
            p.calibration_update()
            p.tic(rgb)
            p.calibration_update()
            p.freeze_calibration()
            state = p.state_dict()
            mu = p.mu.copy()
            p.set_intervention('black')
            p.tic(rgb)
            self.assertEqual(0., np.abs(p.raw_vector()).sum())
            np.testing.assert_array_equal(mu, p.mu)
            self.assertEqual(2, p.calibration_count)
            with self.assertRaises(RuntimeError):
                p.calibration_update()
            q = Pipeline(c)
            q.load_state_dict(state)
            self.assertEqual(0, q._tic)
            self.assertTrue(q.calibrated)
            state['config']['protocol'] = 'old'
            with self.assertRaises(ValueError):
                q.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
