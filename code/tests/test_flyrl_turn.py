"""短程任务纯测试：模拟游戏边界，不读脑图、不运行ViZDoom。"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyrl.environment import LayoutConfig, Telemetry
from flyrl_turn.environment import TurnEnv, TurnGoal, remaining
from flyrl_turn.config import conditions, manifest, load_config


class FakeEngine:
    def __init__(self, owner):
        self.owner = owner
    def is_episode_finished(self):
        return self.owner.finished
    def is_player_dead(self):
        return self.owner.dead


class FakeBoundary:
    def __init__(self, samples, finish_at=None, **kwargs):
        self.samples, self.finish_at = samples, finish_at
        self.tick = 0
        self.finished = self.dead = self.closed = False
        self.actions = []
        self.assets = {'wad_sha256': 'synthetic-test-only'}
        self.game = FakeEngine(self)
    def pixels(self):
        if self.finished:
            raise AssertionError('finished后不得读取RGB')
        return bytes([self.tick, 23, 42])
    def telemetry(self):
        return self.samples[min(self.tick, len(self.samples) - 1)]
    def act(self, action):
        self.actions.append(action.copy())
        self.tick += 1
        self.finished = self.tick == self.finish_at
        self.dead = self.telemetry().health <= 0
        return 99999.  # 原生奖励不得进入任务。
    def close(self):
        self.closed = True


def sample(x=1728., y=128., health=100., angle=0.):
    return Telemetry(x, y, angle, health)


class TurnTests(unittest.TestCase):
    def make_env(self, samples, *, mirror='left', max_tics=280, finish_at=None):
        boundary = FakeBoundary(samples, finish_at)
        env = TurnEnv(7, LayoutConfig(layout_id='near_turn', mirror=mirror), max_tics=max_tics,
                      run_dir='synthetic-unused-assets', game_factory=lambda **kw: boundary)
        self.addCleanup(env.close)
        return env, boundary

    def test_short_distance_and_mirror(self):
        a = LayoutConfig(layout_id='near_turn')
        b = LayoutConfig(layout_id='near_turn', mirror='right')
        self.assertEqual(remaining(1728, 128, a), 576)
        self.assertEqual(remaining(1920, 512, a), 0)
        for x, y in ((1728, 128), (1900, 240), (1920, 400), (1920, 520)):
            self.assertEqual(remaining(x, y, a), remaining(x, -y, b))
        self.assertGreater(remaining(1500, 128, a), 576)

    def test_goal_boundaries_and_wrong_branch(self):
        goal = TurnGoal()
        layout = LayoutConfig(layout_id='near_turn')
        for x, y in ((1792, 512), (2048, 2304), (1920, 800)):
            self.assertTrue(goal.contains(x, y, layout))
        for x, y in ((1791.9, 512), (2048.1, 512), (1920, 511.9), (1920, -512), (1920, 2304.1)):
            self.assertFalse(goal.contains(x, y, layout))
        with self.assertRaises(ValueError):
            TurnGoal(y_min=400)

    def test_success_crosses_goal_once_and_stops(self):
        for mirror, sign in (('left', 1), ('right', -1)):
            with self.subTest(mirror=mirror):
                env, game = self.make_env([sample(y=128*sign), sample(x=1920, y=508*sign),
                                           sample(x=1920, y=520*sign)], mirror=mirror)
                _, start = env.reset()
                self.assertEqual(start['progress'], 0)
                env.step(3)
                rgb, reward, term, trunc, info = env.step(3)
                self.assertTrue(term)
                self.assertFalse(trunc)
                self.assertTrue(info['success'])
                self.assertEqual(info['reason'], 'short_goal')
                self.assertEqual(info['progress'], 1)
                self.assertEqual(info['bootstrap'], 0)
                self.assertEqual(info['reward_terms']['goal'], 10)
                self.assertEqual(rgb, bytes([2, 23, 42]))
                self.assertLess(reward, 11)
                with self.assertRaises(RuntimeError):
                    env.step(3)
                self.assertEqual(game.tick, 2)

    def test_reward_uses_short_goal_and_telescopes(self):
        env, _ = self.make_env([sample(), sample(x=1850), sample(x=1920, y=250), sample(x=1920, y=520)])
        env.reset()
        phi0 = env._phi(sample())
        self.assertEqual(phi0, 0)
        shaping = []
        for _ in range(3):
            *_, info = env.step(3)
            shaping.append(info['reward_terms']['potential'])
        self.assertAlmostEqual(sum(env.gamma**i * r for i, r in enumerate(shaping)), -phi0)
        self.assertTrue(info['success'])
        self.assertNotIn('exit', info['reward_terms'])

    def test_timeout_keeps_true_final_rgb_and_potential(self):
        env, game = self.make_env([sample(), sample(x=1800)], max_tics=1)
        initial, _ = env.reset()
        rgb, r, term, trunc, info = env.step(3)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertNotEqual(rgb, initial)
        self.assertIs(info['final_observation'], rgb)
        self.assertEqual(info['bootstrap'], 1)
        self.assertEqual(info['reward_terms']['goal'], 0)
        expected = env.gamma * env._phi(sample(x=1800))
        self.assertAlmostEqual(info['reward_terms']['potential'], expected)
        self.assertEqual(game.tick, 1)

    def test_success_and_death_precede_timeout(self):
        for health in (100., 0.):
            env, _ = self.make_env([sample(), sample(x=1920, y=520, health=health)], max_tics=1)
            env.reset()
            _, _, term, trunc, info = env.step(3)
            self.assertTrue(term)
            self.assertFalse(trunc)
            self.assertEqual(info['success'], health > 0)
            self.assertEqual(info['reward_terms']['goal'], 10 if health > 0 else 0)

    def test_engine_finish_has_no_rgb_and_no_bootstrap(self):
        env, _ = self.make_env([sample(), sample(x=1800)], finish_at=1)
        env.reset()
        rgb, _, term, trunc, info = env.step(3)
        self.assertIsNone(rgb)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info['bootstrap'], 0)
        self.assertEqual(info['reason'], 'engine_finished_no_final_rgb')

    def test_stationary_and_forward_wall_penalty(self):
        env, game = self.make_env([sample()] * 4)
        env.reset()
        for action in (2, 0):  # 停住或原地转向不算撞墙，不赚正塑形奖励。
            _, r, _, _, info = env.step(action)
            self.assertFalse(info['wall_hit'])
            self.assertEqual(info['progress'], 0)
            self.assertEqual(info['reward_terms']['potential'], 0)
            self.assertAlmostEqual(r, -.001)
        _, r, _, _, info = env.step(3)
        self.assertTrue(info['wall_hit'])
        self.assertAlmostEqual(r, -.021)
        self.assertEqual(game.actions[-1], {'turn': 0., 'forward': 20, 'attack': 0})

    def test_progress_is_episode_relative_and_reset_clears_history(self):
        env, game = self.make_env([sample(), sample(x=1600)])
        rgb, info = env.reset()
        self.assertIsInstance(rgb, bytes)
        self.assertEqual(info['initial_remaining'], 576)
        self.assertEqual(info['progress'], 0)
        *_, info = env.step(3)
        self.assertLess(info['progress'], 0)
        env.reset()
        self.assertTrue(game.closed)  # 旧边界被close；真实factory会创建新实例。
        self.assertEqual(env.tics, 0)

    def test_no_hidden_actions_and_action_validation(self):
        env, game = self.make_env([sample(), sample(x=1730)])
        env.reset()
        self.assertEqual(game.actions, [])
        for invalid in (-1, 6, True, 1.5):
            with self.assertRaises(ValueError):
                env.step(invalid)
        self.assertEqual(game.actions, [])
        env.step(1)
        self.assertEqual(len(game.actions), 1)

    def test_nonfinite_telemetry_rejected(self):
        env, _ = self.make_env([sample(), sample(x=math.nan)])
        env.reset()
        with self.assertRaises(FloatingPointError):
            env.step(3)

    def test_config_and_split_wads_are_prelocked_unique_and_paired(self):
        c = load_config()
        self.assertEqual(c['max_tics'], 280)
        m = manifest()
        hashes = [r['wad_sha256'] for rows in m['splits'].values() for r in rows]
        self.assertEqual(len(hashes), 48)
        self.assertEqual(len(set(hashes)), 48)
        for split, n in (('train', 12), ('dev', 12), ('confirm', 24)):
            rows = conditions(split)
            self.assertEqual(len(rows), n)
            for a, b in zip(rows[::2], rows[1::2]):
                la, lb = LayoutConfig(**a['layout']), LayoutConfig(**b['layout'])
                ax, ay, aa = la.spawn()
                bx, by, ba = lb.spawn()
                self.assertEqual((ax, ay, aa), (bx, -by, -ba % 360))
        with tempfile.TemporaryDirectory() as directory:
            for key, value in (('max_tics', 300), ('action_repeat', True), ('ram_gib', 0), ('extra', 1)):
                path = Path(directory) / 'bad.json'
                path.write_text(json.dumps({key: value}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_spawn_at_goal_or_nonfinite_is_rejected(self):
        for initial in (sample(x=1920, y=520), sample(x=math.nan)):
            env, game = self.make_env([initial])
            with self.assertRaises((ValueError, FloatingPointError)):
                env.reset()
            self.assertTrue(game.closed)

    def test_terminal_failure_does_not_create_positive_shaping_loophole(self):
        env, _ = self.make_env([sample(), sample(x=1920, y=450), sample(x=1920, y=450, health=0)])
        env.reset()
        a = env.step(3)[-1]['reward_terms']['potential']
        b = env.step(2)[-1]['reward_terms']['potential']
        self.assertAlmostEqual(a + env.gamma * b, 0.)

    def test_turn_limits_and_spawn_constraints(self):
        for kw in ({'max_tics': 281}, {'max_tics': True}, {'max_turn': 15},
                   {'layout': LayoutConfig(layout_id='entry')}, {'layout': LayoutConfig(layout_id='near_turn', lightlevel=200)}):
            with self.assertRaises(ValueError):
                TurnEnv(1, **kw)


if __name__ == '__main__':
    unittest.main()
