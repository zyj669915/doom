"""任务验收器和CLI安全测试；全部模拟，不启动真实游戏。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flyrl_turn import acceptance as a
from flyrl_turn.config import load_config
from flyrl_stage import resources

SPEC = importlib.util.spec_from_file_location('turn_cli_under_test', Path(__file__).resolve().parents[1] / 'flyrl_turn_cli.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class AcceptanceTests(unittest.TestCase):
    def test_script_only_uses_time_and_rng_and_random_respects_codec(self):
        self.assertEqual(a.action_at('sequence-negative', 24, None), 0)
        self.assertEqual(a.action_at('sequence-positive', 24, None), 4)
        self.assertEqual(a.action_at('sequence-positive', 40, None), 3)
        for policy in a.POLICIES:
            rng = np.random.default_rng(2)
            self.assertTrue(all(0 <= a.action_at(policy, t, rng) < 6 for t in range(280)))
        with self.assertRaises(ValueError):
            a.action_at('oracle', 0, None)

    def test_policies_apply_to_both_sides_and_complete_report(self):
        def episode(c, path, case, policy, *args):
            success = case['case_id'].startswith('nominal')
            return {'case_id': case['case_id'], 'policy': policy, 'complete': True, 'success': success}
        with tempfile.TemporaryDirectory() as directory, patch.object(a, 'episode', side_effect=episode) as run, \
             patch.object(a, 'source_record', return_value={'synthetic': True}):
            result = a.accept(load_config(), Path(directory), Mock(stop_requested=False))
            self.assertTrue(result['passed'])
            self.assertEqual(run.call_count, 136)
            self.assertEqual(result['completed_episodes'], 136)
            self.assertEqual(result['nominal_reachability'], {'left': True, 'right': True})
            self.assertEqual(set(result['train_open_loop_scores'].values()), {0.})
            self.assertFalse(result['training_started'])

    def test_trivial_policy_rejects_task(self):
        def episode(c, path, case, policy, *args):
            return {'case_id': case['case_id'], 'policy': policy, 'complete': True, 'success': True}
        with tempfile.TemporaryDirectory() as directory, patch.object(a, 'episode', side_effect=episode), \
             patch.object(a, 'source_record', return_value={}):
            result = a.accept(load_config(), Path(directory), Mock(stop_requested=False))
            self.assertFalse(result['passed'])
            self.assertEqual(result['stop'], 'acceptance_failed')

    def test_resource_or_stop_never_runs_episode(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(a, 'episode') as run, \
             patch.object(a, 'source_record', return_value={}):
            result = a.accept(load_config(), Path(directory), Mock(stop_requested=True))
            self.assertFalse(result['complete'])
            self.assertEqual(result['stop'], 'resource_stop')
            run.assert_not_called()

    def test_fake_rgb_episode_records_real_final_frame_and_macro_actions(self):
        class Env:
            tick = 0
            closed = False
            actions = []
            def reset(self):
                return np.zeros((480, 640, 3), np.uint8), {'progress': 0., 'initial_remaining': 576.}
            def step(self, action):
                self.actions.append(action)
                self.tick += 1
                return np.full((480, 640, 3), self.tick, np.uint8), -.001, False, self.tick == 5, {
                    'progress': self.tick / 10, 'reward_terms': {'tic': -.001}, 'success': False,
                    'reason': 'time_limit', 'bootstrap': 1, 'wall_hits': 0, 'telemetry': {}, 'assets': {}}
            def close(self):
                self.closed = True
        env = Env()
        with tempfile.TemporaryDirectory() as directory, patch.object(a, 'make_env', return_value=env):
            case = a.nominal_conditions()[0]
            result = a.episode(load_config(), Path(directory), case, 'random', Mock(stop_requested=False),
                               a.time.monotonic(), 10, {})
            self.assertTrue(result['complete'])
            self.assertTrue(result['truncated'])
            self.assertEqual(result['tics'], 5)
            self.assertEqual(len(set(env.actions[:4])), 1)
            self.assertTrue(env.closed)

    def test_cli_rejected_gate_writes_record_and_never_calls_acceptance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(resources, 'gate', return_value={
            'allowed': False, 'errors': ['synthetic insufficient RAM']}), patch.object(a, 'accept') as run, \
            patch.object(sys, 'argv', ['cli', 'accept', '--output', directory]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(), 3)
            run.assert_not_called()
            result = json.loads(next(Path(directory).glob('*/result.json')).read_text(encoding='utf-8'))
            self.assertFalse(result['started'])
            self.assertEqual(result['stop'], 'resource_gate_rejected')
            self.assertFalse((Path(directory) / 'ACTIVE.lock').exists())

    def test_stop_during_last_episode_vetoes_pass(self):
        count = 0
        def episode(c, path, case, policy, *args):
            nonlocal count
            count += 1
            if count == 136:
                (path / 'STOP').touch()
            return {'case_id': case['case_id'], 'policy': policy, 'complete': True,
                    'success': case['case_id'].startswith('nominal')}
        with tempfile.TemporaryDirectory() as directory, patch.object(a, 'episode', side_effect=episode), \
             patch.object(a, 'source_record', return_value={}):
            result = a.accept(load_config(), Path(directory), Mock(stop_requested=False))
            self.assertTrue(result['complete'])
            self.assertFalse(result['passed'])
            self.assertEqual(result['stop'], 'controlled_stop')

    def test_wall_limit_during_last_episode_vetoes_pass(self):
        count, clock = 0, [0.]
        def episode(c, path, case, policy, *args):
            nonlocal count
            count += 1
            if count == 136:
                clock[0] = 1201.
            return {'case_id': case['case_id'], 'policy': policy, 'complete': True,
                    'success': case['case_id'].startswith('nominal')}
        with tempfile.TemporaryDirectory() as directory, patch.object(a, 'episode', side_effect=episode), \
             patch.object(a, 'source_record', return_value={}), patch.object(a.time, 'monotonic', side_effect=lambda: clock[0]):
            result = a.accept(load_config(), Path(directory), Mock(stop_requested=False))
            self.assertTrue(result['complete'])
            self.assertFalse(result['passed'])
            self.assertEqual(result['stop'], 'wall_limit')

    def test_cli_configuration_is_one_snapshot_even_if_file_changes_after_gate(self):
        class Monitor:
            stop_requested = False
            def __init__(self, path):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text(json.dumps({'ram_gib': 1.5, 'commit_gib': 2.}), encoding='utf-8')
            def gate(ram, commit):
                self.assertEqual((ram, commit), (1.5, 2.))
                config.write_text(json.dumps({'ram_gib': 8., 'commit_gib': 8.}), encoding='utf-8')
                return {'allowed': True, 'errors': []}
            def accept(c, path, monitor, **kwargs):
                self.assertEqual((c['ram_gib'], c['commit_gib']), (1.5, 2.))
                self.assertEqual(c['output'], str(Path(directory) / 'runs'))
                return {'run': str(path), 'passed': True}
            with patch.object(resources, 'gate', side_effect=gate), patch.object(resources, 'MemoryMonitor', Monitor), \
                 patch.object(a, 'accept', side_effect=accept), patch.object(sys, 'argv', [
                     'cli', 'accept', '--config', str(config), '--output', str(Path(directory) / 'runs')]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)
            saved = json.loads(next((Path(directory) / 'runs').glob('*/config.json')).read_text(encoding='utf-8'))
            self.assertEqual((saved['ram_gib'], saved['commit_gib']), (1.5, 2.))

    def test_cli_stop_or_wall_during_monitor_exit_vetoes_pass(self):
        for cause in ('STOP', 'wall'):
            clock = [0.]
            class Monitor:
                stop_requested = False
                def __init__(self, path):
                    self.path = path
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    if cause == 'STOP':
                        (self.path.parent / 'STOP').touch()
                    else:
                        clock[0] = 1201.
                    return False
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory, \
                 patch.object(resources, 'gate', return_value={'allowed': True, 'errors': []}), \
                 patch.object(resources, 'MemoryMonitor', Monitor), \
                 patch.object(a, 'accept', side_effect=lambda c, path, monitor, **kw: {'run': str(path), 'passed': True}), \
                 patch.object(cli.time, 'monotonic', side_effect=lambda: clock[0]), \
                 patch.object(sys, 'argv', ['cli', 'accept', '--output', directory]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 3)
                result = json.loads(next(Path(directory).glob('*/result.json')).read_text(encoding='utf-8'))
                self.assertFalse(result['passed'])
                self.assertEqual(result['stop'], 'controlled_stop' if cause == 'STOP' else 'wall_limit')

    def test_stop_file_and_wall_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'STOP').touch()
            self.assertEqual(a.check_stop(path, Mock(stop_requested=False), a.time.monotonic(), 10), 'controlled_stop')
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(a.check_stop(Path(directory), Mock(stop_requested=False), 0, .01), 'wall_limit')


if __name__ == '__main__':
    unittest.main()
