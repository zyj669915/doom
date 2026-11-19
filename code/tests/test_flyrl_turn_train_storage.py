"""独立 checkpoint 纯测试：仅临时文件、假来源、小 tensor；不加载图/游戏/脑。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from flyrl_turn_train.config import load_config, PROTOCOL
from flyrl_turn_train import storage


class FakeComponent:
    def __init__(self, value):
        self.data = {'weight': torch.tensor([value], dtype=torch.float32),
                     'metadata': {'updates': 3}}

    def state_dict(self):
        return self.data


def fake_provenance():
    return {'sources': {'experiment/code/flyrl_turn_train/storage.py': '1' * 64},
            'inputs': {'graph': {'path': 'test-only/graph.npz', 'sha256': storage.GRAPH_SHA256},
                       'iwad': {'path': 'test-only/freedoom2.wad', 'sha256': '2' * 64}},
            'versions': {name: 'test-only-1' for name in
                         ('python', 'executable', 'numpy', 'torch', 'numba',
                          'vizdoom', 'pandas', 'pyarrow')}}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / 'run'
        self.c = load_config(values={'arm': 'frontend', 'max_game_tics': 32})
        self.prov = fake_provenance()
        self.pipeline, self.agent = FakeComponent(1), FakeComponent(2)
        self.state = {'game_tics': 12, 'decisions': 3, 'counts': {'episodes': 2}}
        self.provenance_patch = patch.object(storage, 'provenance', return_value=deepcopy(self.prov))
        self.current_provenance = self.provenance_patch.start()
        self.addCleanup(self.provenance_patch.stop)
        python_rng, torch_rng = random.getstate(), torch.get_rng_state()
        self.addCleanup(random.setstate, python_rng)
        self.addCleanup(torch.set_rng_state, torch_rng)

    def save(self):
        return storage.save_checkpoint(self.run, self.c, self.prov,
                                       self.pipeline, self.agent, self.state)

    def read_json(self, path):
        return json.loads(path.read_text(encoding='utf-8'))

    def rewrite_manifest(self, path, mutate):
        manifest = self.read_json(path / 'manifest.json')
        mutate(manifest)
        storage.write_json(path / 'manifest.json', manifest)

    def assert_rejected_before_torch(self, path, config=None):
        with patch.object(torch, 'load') as unsafe_load:
            with self.assertRaises((ValueError, FileNotFoundError)):
                storage.load_checkpoint(path, config)
            unsafe_load.assert_not_called()

    def assert_only_old_checkpoint_committed(self, old, expected_incomplete=1):
        self.assertEqual(self.read_json(self.run / 'latest-checkpoint.json'),
                         {'checkpoint': str(old)})
        self.assertEqual([p for p in self.run.glob('checkpoint-*')
                          if not p.name.endswith('.incomplete')], [old])
        self.assertEqual(len(list(self.run.glob('*.incomplete'))), expected_incomplete)

    def test_atomic_success_manifest_and_safe_cpu_load(self):
        with patch.object(storage.os, 'replace', wraps=storage.os.replace) as replace:
            path = self.save()
        self.assertTrue(path.is_dir())
        self.assertRegex(path.name, r'^checkpoint-000000012-[0-9a-f]{32}$')
        self.assertFalse(list(self.run.glob('*.incomplete')))
        self.assertEqual(self.read_json(self.run / 'latest-checkpoint.json'),
                         {'checkpoint': str(path)})
        transitions = [(Path(call.args[0]), Path(call.args[1])) for call in replace.call_args_list]
        self.assertIn((path.with_name(path.name + '.incomplete'), path), transitions)
        with patch.object(torch, 'load', wraps=torch.load) as safe_load:
            saved, manifest, data = storage.load_checkpoint(path, self.c)
        self.assertEqual(saved, self.c)
        self.assertEqual(manifest['schema_version'], 1)
        self.assertEqual(manifest['protocol'], PROTOCOL)
        self.assertEqual(manifest['state_sha256'], storage.sha256(path / 'state.pt'))
        self.assertEqual(manifest['config_sha256'], storage._config_sha256(self.c))
        self.assertIn('not exact game restoration', manifest['resume_semantics'])
        self.assertEqual(data['state'], self.state)
        torch.testing.assert_close(data['pipeline']['weight'], self.pipeline.data['weight'])
        torch.testing.assert_close(data['learner']['weight'], self.agent.data['weight'])
        self.assertEqual(safe_load.call_args.kwargs, {'map_location': 'cpu', 'weights_only': True})

    def test_read_manifest_is_json_only_and_does_not_probe_inputs(self):
        path = self.save()
        self.current_provenance.reset_mock()
        with patch.object(torch, 'load') as tensor_load, patch.object(storage, 'sha256') as hash_file:
            manifest = storage.read_manifest(path)
            self.assertEqual(storage.read_manifest(path / 'manifest.json'), manifest)
        tensor_load.assert_not_called()
        hash_file.assert_not_called()
        self.current_provenance.assert_not_called()

    def test_state_save_failure_keeps_incomplete_and_does_not_update_latest(self):
        old = self.save()
        with patch.object(torch, 'save', side_effect=RuntimeError('test-only save failure')):
            with self.assertRaisesRegex(RuntimeError, 'test-only'):
                self.save()
        self.assert_only_old_checkpoint_committed(old)
        incomplete = next(self.run.glob('*.incomplete'))
        self.assertFalse((incomplete / 'manifest.json').exists())

    def test_first_save_failure_does_not_create_latest_or_final(self):
        with patch.object(torch, 'save', side_effect=RuntimeError('first save failure')):
            with self.assertRaises(RuntimeError):
                self.save()
        self.assertFalse((self.run / 'latest-checkpoint.json').exists())
        self.assertEqual(len(list(self.run.glob('*.incomplete'))), 1)
        self.assertFalse([p for p in self.run.glob('checkpoint-*')
                          if not p.name.endswith('.incomplete')])

    def test_state_dict_failure_keeps_incomplete_and_latest(self):
        old = self.save()
        with patch.object(self.agent, 'state_dict', side_effect=RuntimeError('state_dict failure')):
            with self.assertRaises(RuntimeError):
                self.save()
        self.assert_only_old_checkpoint_committed(old)

    def test_manifest_failure_keeps_incomplete_and_latest(self):
        old = self.save()
        with patch.object(storage, 'write_json', side_effect=OSError('manifest failure')):
            with self.assertRaises(OSError):
                self.save()
        self.assert_only_old_checkpoint_committed(old)

    def test_latest_failure_rolls_back_final_and_preserves_old_pointer(self):
        old = self.save()
        original = storage.write_json

        def fail_latest(path, value):
            if Path(path).name == 'latest-checkpoint.json':
                raise OSError('latest failure')
            return original(path, value)

        with patch.object(storage, 'write_json', side_effect=fail_latest):
            with self.assertRaises(OSError):
                self.save()
        self.assert_only_old_checkpoint_committed(old)
        self.assertTrue((next(self.run.glob('*.incomplete')) / 'manifest.json').exists())

    def test_save_rechecks_all_provenance_and_keeps_failed_evidence(self):
        old = self.save()
        for key in ('sources', 'inputs', 'versions'):
            with self.subTest(part=key):
                current = deepcopy(self.prov)
                if key == 'sources':
                    current[key]['experiment/code/flyrl_turn_train/storage.py'] = '3' * 64
                elif key == 'inputs':
                    current[key]['iwad']['sha256'] = '4' * 64
                else:
                    current[key]['torch'] = 'changed'
                self.current_provenance.return_value = current
                with self.assertRaisesRegex(ValueError, '漂移'):
                    self.save()
        self.assert_only_old_checkpoint_committed(old, expected_incomplete=3)
        for path in self.run.glob('*.incomplete'):
            self.assertTrue((path / 'state.pt').is_file())
            self.assertFalse((path / 'manifest.json').exists())

    def test_unexpected_provenance_error_never_commits(self):
        old = self.save()
        self.current_provenance.side_effect = OSError('unavailable source')
        with self.assertRaises(OSError):
            self.save()
        self.assert_only_old_checkpoint_committed(old)

    def test_snapshots_are_not_live_references(self):
        expected_c, expected_prov, expected_state = map(deepcopy, (self.c, self.prov, self.state))
        expected_pipeline, expected_agent = deepcopy(self.pipeline.data), deepcopy(self.agent.data)
        original = torch.save

        def mutate_callers_then_save(data, stream):
            self.c['seed'] += 1
            self.prov['versions']['torch'] = 'mutated'
            self.state['counts']['episodes'] = 999
            self.pipeline.data['weight'].add_(90)
            self.agent.data['metadata']['updates'] = 999
            return original(data, stream)

        with patch.object(torch, 'save', side_effect=mutate_callers_then_save):
            path = self.save()
        saved, manifest, data = storage.load_checkpoint(path)
        self.assertEqual(saved, expected_c)
        self.assertEqual(manifest['provenance'], expected_prov)
        self.assertEqual(data['state'], expected_state)
        torch.testing.assert_close(data['pipeline']['weight'], expected_pipeline['weight'])
        self.assertEqual(data['learner']['metadata'], expected_agent['metadata'])

    def test_state_hash_drift_rejected_before_load(self):
        path = self.save()
        with (path / 'state.pt').open('ab') as stream:
            stream.write(b'test-only tamper')
        self.current_provenance.reset_mock()
        self.assert_rejected_before_torch(path)
        self.current_provenance.assert_not_called()

    def test_source_drift_addition_and_deletion_rejected_before_load(self):
        path = self.save()
        for change in ('hash', 'add', 'delete'):
            with self.subTest(change=change):
                prov = deepcopy(self.prov)
                if change == 'hash':
                    prov['sources']['experiment/code/flyrl_turn_train/storage.py'] = '5' * 64
                elif change == 'add':
                    prov['sources']['experiment/code/flyrl_turn_train/new.py'] = '6' * 64
                else:
                    prov['sources'].clear()
                self.current_provenance.return_value = prov
                self.assert_rejected_before_torch(path)

    def test_graph_and_asset_hash_or_path_drift_rejected_before_load(self):
        path = self.save()
        for key in ('graph', 'iwad'):
            for field in ('path', 'sha256'):
                with self.subTest(input=key, field=field):
                    prov = deepcopy(self.prov)
                    prov['inputs'][key][field] = 'different/path' if field == 'path' else '7' * 64
                    self.current_provenance.return_value = prov
                    self.assert_rejected_before_torch(path)

    def test_python_executable_and_all_dependency_version_drift_rejected(self):
        path = self.save()
        for name in self.prov['versions']:
            with self.subTest(version=name):
                prov = deepcopy(self.prov)
                prov['versions'][name] += '-changed'
                self.current_provenance.return_value = prov
                self.assert_rejected_before_torch(path)

    def test_manifest_config_hash_or_state_hash_fields_rejected(self):
        path = self.save()
        original = self.read_json(path / 'manifest.json')
        for key, value in (('config_sha256', '0' * 64), ('state_sha256', '0' * 64),
                           ('config_sha256', 'not-a-sha'), ('state_sha256', None)):
            with self.subTest(field=key, value=value):
                manifest = deepcopy(original)
                manifest[key] = value
                storage.write_json(path / 'manifest.json', manifest)
                self.assert_rejected_before_torch(path)

    def test_invalid_schema_protocol_config_or_resume_semantics_rejected(self):
        path = self.save()
        original = self.read_json(path / 'manifest.json')
        variants = []
        for field, value in (('schema_version', 2), ('schema_version', True),
                             ('protocol', 'malecns-near-turn-three-arm-v1'),
                             ('config', []), ('resume_semantics', 'exact game restoration')):
            manifest = deepcopy(original)
            manifest[field] = value
            variants.append(manifest)
        for mutation in ('protocol', 'environment_protocol', 'arm', 'missing'):
            manifest = deepcopy(original)
            if mutation == 'missing':
                del manifest['config']['seed']
            elif mutation == 'arm':
                manifest['config']['arm'] = 'rewired'
            else:
                manifest['config'][mutation] = 'old-protocol'
            manifest['config_sha256'] = storage._config_sha256(manifest['config'])
            variants.append(manifest)
        for index, manifest in enumerate(variants):
            with self.subTest(variant=index):
                storage.write_json(path / 'manifest.json', manifest)
                self.assert_rejected_before_torch(path)

    def test_legacy_stage_manifest_rejected_before_tensor_load(self):
        path = self.save()
        legacy = {'config': {'protocol': 'malecns-near-turn-three-arm-v1'},
                  'provenance': self.prov, 'sha256': storage.sha256(path / 'state.pt'),
                  'resume_semantics': 'new_episode'}
        storage.write_json(path / 'manifest.json', legacy)
        self.assert_rejected_before_torch(path)

    def test_resume_allows_only_three_budget_or_location_fields(self):
        path = self.save()
        before = (path / 'manifest.json').read_bytes()
        requested = load_config(values={**self.c, 'output': 'different/output',
                                        'max_game_tics': 64, 'wall_hours': 8.})
        saved, _, _ = storage.load_checkpoint(path, requested)
        self.assertEqual(saved, self.c)
        self.assertEqual((path / 'manifest.json').read_bytes(), before)

    def test_arm_environment_seed_and_hyperparameter_mixing_rejected(self):
        path = self.save()
        changes = ({'arm': 'real'}, {'seed': self.c['seed'] + 1}, {'projection_seed': 99},
                   {'environment_protocol': 'other-environment'}, {'protocol': 'old-protocol'},
                   {'epsilon_end': .2}, {'calibration_steps': 64}, {'ram_gib': 2.},
                   {'commit_gib': 3.}, {'eval_wall_hours': 1.5},
                   {'graph': 'different/graph.npz'}, {'learner': {'lr': .001}})
        for change in changes:
            with self.subTest(change=change):
                self.assert_rejected_before_torch(path, {**self.c, **change})

    def test_rng_content_can_be_restored(self):
        random.seed(12345)
        torch.manual_seed(54321)
        python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        path = self.save()
        expected_python, expected_torch = random.random(), torch.rand(4)
        random.seed(9)
        torch.manual_seed(9)
        _, _, data = storage.load_checkpoint(path)
        self.assertEqual(data['python_rng'], python_rng)
        torch.testing.assert_close(data['torch_rng'], torch_rng)
        random.setstate(data['python_rng'])
        torch.set_rng_state(data['torch_rng'])
        self.assertEqual(random.random(), expected_python)
        torch.testing.assert_close(torch.rand(4), expected_torch)

    def test_checkpoint_write_wall_is_recorded_and_restored_exactly_once(self):
        self.state['wall_s'] = 10.
        with patch.object(storage.time, 'monotonic', side_effect=[100., 102.5]):
            path = self.save()
        self.assertEqual(self.state['wall_s'], 10.)
        before_manifest = (path / 'manifest.json').read_bytes()
        before_state = (path / 'state.pt').read_bytes()
        for _ in range(2):
            _, manifest, data = storage.load_checkpoint(path)
            self.assertEqual(manifest['checkpoint_write_seconds'], 2.5)
            self.assertEqual(data['state']['wall_s'], 12.5)
        self.assertEqual((path / 'manifest.json').read_bytes(), before_manifest)
        self.assertEqual((path / 'state.pt').read_bytes(), before_state)

    def test_invalid_checkpoint_write_wall_rejected_before_load(self):
        path = self.save()
        original = self.read_json(path / 'manifest.json')
        for value in (None, True, -1., '2.5'):
            with self.subTest(value=value):
                manifest = deepcopy(original)
                manifest['checkpoint_write_seconds'] = value
                storage.write_json(path / 'manifest.json', manifest)
                self.assert_rejected_before_torch(path)

    def test_directory_commit_failure_does_not_update_latest(self):
        old = self.save()
        original = storage.os.replace

        def fail_directory(source, destination):
            if Path(source).name.endswith('.incomplete'):
                raise OSError('directory commit failure')
            return original(source, destination)

        with patch.object(storage.os, 'replace', side_effect=fail_directory):
            with self.assertRaises(OSError):
                self.save()
        self.assert_only_old_checkpoint_committed(old)

    def test_incomplete_checkpoint_is_never_loadable(self):
        path = self.save()
        incomplete = path.with_name(path.name + '.incomplete')
        path.rename(incomplete)
        self.assert_rejected_before_torch(incomplete)

    def test_rejected_checkpoint_never_rewrites_historical_manifest(self):
        path = self.save()
        before = (path / 'manifest.json').read_bytes()
        self.current_provenance.return_value = {**self.prov, 'unexpected': True}
        self.assert_rejected_before_torch(path)
        self.assertEqual((path / 'manifest.json').read_bytes(), before)

    def test_duplicate_json_fields_and_nan_rejected(self):
        path = self.save()
        original = (path / 'manifest.json').read_text(encoding='utf-8')
        for prefix in ('"schema_version":1,', '"untrusted":NaN,'):
            with self.subTest(prefix=prefix):
                (path / 'manifest.json').write_text('{' + prefix + original[1:], encoding='utf-8')
                self.assert_rejected_before_torch(path)


class JsonAndHashTests(unittest.TestCase):
    def test_json_atomic_failure_preserves_existing_and_creates_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'record.json'
            storage.write_json(path, {'中文': 1})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                storage.write_json(path, {'bad': float('nan')})
            self.assertEqual(path.read_bytes(), before)
            with patch.object(storage.os, 'replace', side_effect=OSError('replace failure')):
                with self.assertRaises(OSError):
                    storage.write_json(path, {'replacement': 2})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_jsonl_finite_complete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'rows.jsonl'
            storage.append_json(path, {'step': 1, 'label': '中文'})
            storage.append_json(path, {'step': 2})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                storage.append_json(path, {'bad': float('inf')})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()],
                             [{'step': 1, 'label': '中文'}, {'step': 2}])

    def test_sha256_streams_multiple_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.bin'
            payload = b'test-only' * (1 << 18)
            path.write_bytes(payload)
            self.assertEqual(storage.sha256(path), hashlib.sha256(payload).hexdigest())


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for folder in storage._SOURCE_FOLDERS:
            path = self.root / folder
            path.mkdir(parents=True)
            (path / 'module.py').write_text('# test-only source\n', encoding='utf-8')
        nested = self.root / storage._SOURCE_FOLDERS[0] / 'nested'
        nested.mkdir()
        (nested / 'extra.py').write_text('# recursive test-only source\n', encoding='utf-8')
        for filename in storage._SOURCE_FILES:
            (self.root / 'experiment/code' / filename).write_text('# wrapper\n', encoding='utf-8')
        self.graph = self.root / 'graph.npz'
        self.graph.write_bytes(b'test-only graph; never parsed')
        iwad = self.root / 'vizdoom/freedoom2.wad'
        iwad.parent.mkdir()
        iwad.write_bytes(b'test-only asset; never executed')
        self.iwad = iwad
        distribution = SimpleNamespace(files=[Path('vizdoom/freedoom2.wad')],
                                       locate_file=lambda entry: self.root / entry)
        for mocker in (patch.object(storage, 'ROOT', self.root),
                       patch.object(storage, 'GRAPH_SHA256', storage.sha256(self.graph)),
                       patch.object(storage.importlib.metadata, 'distribution', return_value=distribution),
                       patch.object(storage.importlib.metadata, 'version', side_effect=lambda name: name + '-test')):
            mocker.start()
            self.addCleanup(mocker.stop)
        self.c = load_config(values={'arm': 'frontend', 'graph': str(self.graph)})

    def test_recursive_sources_wrappers_versions_and_metadata_asset(self):
        forbidden = SimpleNamespace(DoomGame=lambda: self.fail('不能实例化游戏'))
        with patch.dict(sys.modules, {'vizdoom': forbidden, 'flyrl.brain': None}):
            prov = storage.provenance(self.c)
        self.assertEqual(len(prov['sources']), len(storage._SOURCE_FOLDERS) + len(storage._SOURCE_FILES) + 1)
        self.assertIn('experiment/code/flyrl_turn_train/nested/extra.py', prov['sources'])
        for folder in storage._SOURCE_FOLDERS:
            self.assertIn(folder + '/module.py', prov['sources'])
        for name in storage._SOURCE_FILES:
            self.assertIn('experiment/code/' + name, prov['sources'])
        self.assertTrue(all('\\' not in key and not Path(key).is_absolute() for key in prov['sources']))
        self.assertEqual(prov['versions']['python'], sys.version)
        self.assertEqual(prov['versions']['executable'], sys.executable)
        for name in ('numpy', 'torch', 'numba', 'vizdoom', 'pandas', 'pyarrow'):
            self.assertEqual(prov['versions'][name], name + '-test')
        self.assertEqual(prov['inputs']['graph']['sha256'], storage.GRAPH_SHA256)
        self.assertEqual(prov['inputs']['iwad']['sha256'], storage.sha256(self.iwad))
        self.assertEqual(set(prov['inputs']), {'graph', 'iwad'})
        storage._validate_provenance(prov)

    def test_frontend_rejects_non_authoritative_graph_without_loading_brain(self):
        self.graph.write_bytes(b'changed graph')
        with patch.dict(sys.modules, {'flyrl.brain': None}), self.assertRaisesRegex(ValueError, '权威图'):
            storage.provenance(self.c)

    def test_missing_required_source_is_not_silently_omitted(self):
        (self.root / 'experiment/code/flyrl_turn_cli.py').unlink()
        with self.assertRaises(FileNotFoundError):
            storage.provenance(self.c)

    def test_added_nested_source_changes_provenance(self):
        before = storage.provenance(self.c)
        path = self.root / 'doomfly/doom/nested/new.py'
        path.parent.mkdir()
        path.write_text('# added source\n', encoding='utf-8')
        after = storage.provenance(self.c)
        self.assertNotEqual(before, after)
        self.assertIn('doomfly/doom/nested/new.py', after['sources'])


if __name__ == '__main__':
    unittest.main()
