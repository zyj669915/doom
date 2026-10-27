"""宏动作真实时间和原子快照边界的纯测试。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyrl_next.runner import macro_step, actions, checkpoint, load_checkpoint


class Pipe:
    last_diagnostic = {'actual_ms': 28.5}
    def __init__(self):
        self.calls = 0
    def tic(self, rgb):
        self.calls += 1
    def observation(self):
        return {'vector': np.array([self.calls], np.float32)}


class Env:
    _telemetry = None
    def __init__(self, terminal=False):
        self.i = 0
        self.terminal = terminal
    def _info(self, _):
        return {'progress': self.i / 10}
    def step(self, action):
        self.i += 1
        end = self.i == 2
        return np.zeros((4, 4, 3), np.uint8), 1., end and self.terminal, end and not self.terminal, {'progress': self.i / 10, 'reward_terms': {'tic': 1.}}


class RunnerTests(unittest.TestCase):
    def config(self):
        return {'action_repeat': 4, 'gamma_tic': .9, 'objective': 'potential', 'task': 'corridor', 'action_set': 'full', 'max_turn': 6}
    def test_actual_duration_and_truncation(self):
        p = Pipe()
        obs, r, d, term, trunc, info = macro_step(Env(), p, 0, self.config())
        self.assertAlmostEqual(r, 1.9)
        self.assertAlmostEqual(d, .81)
        self.assertEqual(p.calls, 2)
        self.assertAlmostEqual(info['brain_ms'], 57.)
        self.assertTrue(trunc)
    def test_terminal_and_remaining(self):
        self.assertEqual(macro_step(Env(True), Pipe(), 0, self.config())[2], 0.)
        out = macro_step(Env(), Pipe(), 0, self.config(), remaining=1)
        self.assertEqual(out[-1]['macro_tics'], 1)
        self.assertAlmostEqual(out[2], .9)
    def test_forward_codec(self):
        c = self.config()
        c['action_set'] = 'forward'
        self.assertEqual(actions(c), [1, 3, 5])
        c['task'] = 'combat'
        self.assertEqual(actions(c), [2, 3, 6, 7, 10, 11])
    def test_atomic_failure_not_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            class State:
                def state_dict(self):
                    return {}
            with patch('flyrl_next.runner.torch.save', side_effect=OSError('模拟写盘失败')):
                with self.assertRaises(OSError):
                    checkpoint(Path(directory), {}, {}, State(), State(), {'game_tics': 1})
            self.assertFalse((Path(directory) / 'latest-checkpoint.json').exists())


if __name__ == '__main__':
    unittest.main()
