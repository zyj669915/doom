"""Pure configuration/curriculum tests: no engine, graph or training."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

from flyrl_next.config import Curriculum, ROOT, ROUTE_ALGORITHMS, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_complete_and_detached(self):
        c = load_config()
        self.assertEqual(ROOT, Path(__file__).resolve().parents[3])
        self.assertEqual((len(c['val_layouts']), len(c['test_layouts'])), (12, 24))
        self.assertEqual(c['action_repeat'], 4)
        self.assertEqual(c['device'], 'cpu')
        self.assertEqual(c['rollout_steps'], 256)
        c['train_layouts'][0]['layout_seed'] = 999
        c['curriculum_gate']['window'] = 1
        self.assertNotEqual(load_config()['train_layouts'][0]['layout_seed'], 999)
        self.assertEqual(load_config()['curriculum_gate']['window'], 20)

    def test_route_algorithm(self):
        for route, algorithm in ROUTE_ALGORITHMS.items():
            self.assertEqual(load_config(overrides={'route': route})['algorithm'], algorithm)
            with self.assertRaises(ValueError):
                load_config(overrides={'route': route, 'algorithm': 'ppo' if algorithm == 'dqn' else 'dqn'})

    def test_strict_unknown_fields(self):
        for override in ({'typo': 1}, {'learner': {'typo': 1}}, {'rewards': {'typo': 1}},
                         {'curriculum_gate': {'typo': 1}}, {'learner': {'gamma': .5}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(overrides=override)

    def test_learner_uses_real_dataclass_fields(self):
        self.assertEqual(load_config(overrides={'learner': {'target_every': 42}})['learner']['target_every'], 42)
        for route, bad in (('A3', 'target_sync_interval'), ('A3', 'epsilon_end'),
                           ('A4', 'rollout_steps'), ('A4', 'critic_hidden')):
            with self.subTest(route=route, key=bad), self.assertRaises(ValueError):
                load_config(overrides={'route': route, 'learner': {bad: 1}})

    def test_invalid_scalar_values(self):
        for override in ({'threshold': -53}, {'device': 'mps'}, {'wall_hours': float('nan')},
                         {'gamma_tic': 1.1}, {'action_repeat': True}, {'calibration_steps': 1},
                         {'seed': -1}, {'hidden': 64}, {'learner': {'lr': False}},
                         {'curriculum_gate': {'window': 5}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(overrides=override)

    def test_seed_duplicates_and_overlap(self):
        for override in ({'train_seeds': [1, 1]}, {'train_seeds': [51027]}, {'test_seeds': [True]}):
            with self.assertRaises(ValueError):
                load_config(overrides=override)

    def test_real_wad_duplicate_with_different_seed(self):
        c = load_config()
        # When jitter is disabled, distinct metadata seeds produce identical WADs.
        items = c['val_layouts']
        items[0].update(spawn_jitter=0, angle_jitter=0)
        items[2] = {**items[0], 'layout_seed': 123456}
        with self.assertRaisesRegex(ValueError, 'WAD'):
            load_config(overrides={'val_layouts': items})

    def test_cross_split_wad_duplicates(self):
        c = load_config()
        c['val_layouts'][0] = deepcopy(c['train_layouts'][0])
        with self.assertRaisesRegex(ValueError, 'WAD'):
            load_config(overrides=c)

    def test_nested_merge_and_file_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'route': 'A3', 'rewards': {'exit_bonus': 20}}), encoding='utf-8')
            c = load_config(path, {'wall_hours': 8, 'rewards': {'tic_cost': -.002}})
        self.assertEqual(c['wall_hours'], 8)
        self.assertEqual(c['rewards']['exit_bonus'], 20)
        self.assertEqual(c['rewards']['tic_cost'], -.002)
        self.assertIn('wall_penalty', c['rewards'])

    def test_suite_manifests_reference_configs_and_fail_closed_resources(self):
        directory = ROOT / 'experiment/configs_next'
        for path in directory.glob('suite*.json'):
            manifest = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(set(manifest), {'output', 'runs'})
            self.assertEqual(len({r['id'] for r in manifest['runs']}), len(manifest['runs']))
            for run in manifest['runs']:
                self.assertTrue((directory / run['config']).is_file())
                self.assertEqual(run['resources'], {'ram_gib': None, 'commit_gib': None, 'vram_gib': 0})

    def test_all_shipped_configs(self):
        configs = sorted((ROOT / 'experiment/configs_next').glob('A*.json'))
        self.assertGreaterEqual(len(configs), 22)
        for path in configs:
            with self.subTest(path=path.name):
                c = load_config(path)
                self.assertEqual(c['hidden'], 0)
                self.assertEqual(c['algorithm'], ROUTE_ALGORITHMS[c['route']])


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config()
        self.curriculum = Curriculum(self.c)

    def layouts(self, stage):
        return [x for x in self.c['train_layouts'] if x['layout_id'] == stage]

    def test_success_gate_needs_both_sides_and_minimum_episodes(self):
        left, right = self.layouts('exit')[:2]
        for _ in range(10):
            self.curriculum.record(left, True)
        self.assertEqual(self.curriculum.stage, 0)
        self.curriculum.record(right, True)
        self.assertEqual(self.curriculum.stage, 0)
        self.curriculum.record(right, True)
        self.assertEqual(self.curriculum.stage, 1)

    def test_failures_do_not_advance(self):
        for i in range(50):
            self.curriculum.record(self.layouts('exit')[i % 2], False)
        self.assertEqual(self.curriculum.stage, 0)
        self.assertEqual(len(self.curriculum.state_dict()['history']['exit']), 20)

    def test_progression_mix_and_restore(self):
        for stage in ('exit', 'near_turn'):
            for i in range(10):
                self.curriculum.record(self.layouts(stage)[i % 2], True)
        self.assertEqual(self.curriculum.stage, 2)
        rng = random.Random(1)
        stages = [self.curriculum.select(rng)['layout_id'] for _ in range(1000)]
        self.assertTrue(.74 < stages.count('entry') / 1000 < .86)
        self.assertIn('exit', stages)
        self.assertIn('near_turn', stages)
        state = self.curriculum.state_dict()
        other = Curriculum(self.c)
        other.load_state_dict(state)
        self.assertEqual(other.state_dict(), state)
        state['history']['entry'].append({'mirror': 'left', 'success': True})
        self.assertNotEqual(other.state_dict(), state)

    def test_only_train_feedback(self):
        with self.assertRaises(ValueError):
            self.curriculum.record(self.c['val_layouts'][0], True)
        with self.assertRaises(ValueError):
            self.curriculum.record(self.layouts('entry')[0], True)

    def test_fixed_original_proportional_schedule_and_restore(self):
        config = load_config(overrides={'curriculum': 'fixed', 'max_game_tics': 1000})
        curriculum = Curriculum(config)
        rng = random.Random(1)
        for increment, expected in ((0, 'exit'), (99, 'exit'), (1, 'near_turn'),
                                    (199, 'near_turn'), (1, 'entry')):
            curriculum.advance(increment)
            for _ in range(20):
                self.assertEqual(curriculum.select(rng)['layout_id'], expected)
            restored = Curriculum(config)
            restored.load_state_dict(curriculum.state_dict())
            self.assertEqual(restored.state_dict(), curriculum.state_dict())
        bad = curriculum.state_dict()
        bad['stage'] = 0
        with self.assertRaises(ValueError):
            curriculum.load_state_dict(bad)

    def test_restore_allows_wall_extension_but_rejects_tic_budget_change(self):
        state = self.curriculum.state_dict()
        extended_wall = Curriculum(load_config(overrides={'wall_hours': 8}))
        extended_wall.load_state_dict(state)
        changed_tics = Curriculum(load_config(overrides={'max_game_tics': 20000000}))
        with self.assertRaisesRegex(ValueError, 'max_game_tics'):
            changed_tics.load_state_dict(state)

    def test_bad_state_is_atomic(self):
        before = self.curriculum.state_dict()
        state = deepcopy(before)
        state['history']['entry'] = [{'mirror': 'left', 'success': 1}]
        with self.assertRaises(ValueError):
            self.curriculum.load_state_dict(state)
        self.assertEqual(self.curriculum.state_dict(), before)


if __name__ == '__main__':
    unittest.main()
