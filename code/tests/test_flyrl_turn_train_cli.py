"""新CLI的门禁、单快照与最终停止回归；替身runner不运行游戏。"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from flyrl_stage import resources
from flyrl_turn_train.config import ROOT

SPEC = importlib.util.spec_from_file_location('turn_train_cli',
    Path(__file__).resolve().parents[1] / 'flyrl_turn_train_cli.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class FakeMonitor:
    def __init__(self, path):
        self.path = path
        self.stop_requested = False
        self.on_exit = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.on_exit:
            self.on_exit(self)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / 'runs'
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'arm': 'frontend', 'ram_gib': 1.5, 'commit_gib': 2.}), encoding='utf-8')
        self.addCleanup(self.tmp.cleanup)
        self.stdout = io.StringIO()
        self.fake_runner = types.ModuleType('flyrl_turn_train.runner')
        self.fake_runner.run_train = lambda c, path, monitor, **kw: {
            'run': str(path), 'complete': True, 'training_started': False}
        self.fake_runner.run_evaluate = self.fake_runner.run_train
        self.patches = [patch.object(cli, 'ROOT', self.root),
                        patch.object(resources, 'gate', return_value={'allowed': True, 'errors': []}),
                        patch.object(resources, 'MemoryMonitor', FakeMonitor),
                        patch.dict(sys.modules, {'flyrl_turn_train.runner': self.fake_runner})]
        (self.root / 'experiment').mkdir()
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def invoke(self, command='train', extra=()):
        with redirect_stdout(self.stdout):
            return cli.main([command, '--config', str(self.config), '--output', str(self.output), *extra])

    def results(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in self.output.glob('*/result.json')]

    def test_gate_rejection_never_imports_heavy_runner(self):
        with patch.object(resources, 'gate', return_value={'allowed': False, 'errors': ['low RAM']}), \
             patch.dict(sys.modules, {'flyrl_turn_train.runner': None}):
            self.assertEqual(3, self.invoke())
        self.assertEqual('resource_gate_rejected', self.results()[0]['stop'])
        self.assertFalse((self.root / 'experiment/TURN_TRAIN_ACTIVE.lock').exists())

    def test_one_config_snapshot_even_if_gate_changes_original(self):
        captured = {}
        def gate(ram, commit):
            self.config.write_text(json.dumps({'arm': 'real', 'ram_gib': 8., 'commit_gib': 8.}), encoding='utf-8')
            return {'allowed': True, 'errors': []}
        def run(c, path, monitor, **kw):
            captured.update(c)
            return {'run': str(path), 'complete': True}
        self.fake_runner.run_train = run
        with patch.object(resources, 'gate', side_effect=gate):
            self.assertEqual(0, self.invoke())
        self.assertEqual('frontend', captured['arm'])
        self.assertEqual(1.5, captured['ram_gib'])
        saved = json.loads(next(self.output.glob('*/config.json')).read_text(encoding='utf-8'))
        self.assertEqual(captured, saved)
        self.assertEqual(str(self.output), saved['output'])

    def test_monitor_exit_stop_vetoes_complete_and_releases_lock(self):
        class ExitMonitor(FakeMonitor):
            def __exit__(inner, *_):
                (inner.path.parent / 'STOP').touch()
        with patch.object(resources, 'MemoryMonitor', ExitMonitor):
            self.assertEqual(3, self.invoke())
        self.assertEqual('controlled_stop', self.results()[0]['stop'])
        self.assertFalse(self.results()[0]['complete'])
        self.assertFalse((self.root / 'experiment/TURN_TRAIN_ACTIVE.lock').exists())

    def test_monitor_exit_resource_stop_vetoes_complete(self):
        class ExitMonitor(FakeMonitor):
            def __exit__(inner, *_):
                inner.stop_requested = True
        with patch.object(resources, 'MemoryMonitor', ExitMonitor):
            self.assertEqual(3, self.invoke())
        self.assertEqual('resource_stop', self.results()[0]['stop'])

    def test_wall_limit_includes_dependency_import_and_final_exit(self):
        clock = [0.]
        class ExitMonitor(FakeMonitor):
            def __exit__(inner, *_):
                clock[0] = 21601.
        with patch.object(resources, 'MemoryMonitor', ExitMonitor), \
             patch.object(cli.time, 'monotonic', side_effect=lambda: clock[0]):
            self.assertEqual(3, self.invoke())
        self.assertEqual('wall_limit', self.results()[0]['stop'])

    def test_fixed_slot_does_not_delete_unknown_lock(self):
        lock = self.root / 'experiment/TURN_TRAIN_ACTIVE.lock'
        lock.write_text('owned by another process', encoding='utf-8')
        self.assertEqual(3, self.invoke())
        self.assertEqual('owned by another process', lock.read_text(encoding='utf-8'))
        self.assertEqual('active_slot_locked', self.results()[0]['stop'])

    def test_exception_saves_failure_and_releases_owned_lock(self):
        def failure(*args, **kwargs):
            raise ValueError('intentional fake failure')
        self.fake_runner.run_train = failure
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertTrue(list(self.output.glob('*/failure.json')))
        self.assertFalse((self.root / 'experiment/TURN_TRAIN_ACTIVE.lock').exists())

    def test_final_eval_stop_also_nulls_success_rate(self):
        class ExitMonitor(FakeMonitor):
            def __exit__(inner, *_):
                (inner.path.parent / 'STOP').touch()
        self.fake_runner.run_evaluate = lambda c, path, monitor, *a, **kw: {
            'run': str(path), 'complete': True, 'success_rate': .75, 'episodes': [1, 2]}
        with patch.object(resources, 'MemoryMonitor', ExitMonitor):
            self.assertEqual(3, self.invoke('evaluate', ('--checkpoint', 'test-only')))
        self.assertIsNone(self.results()[0]['success_rate'])
        self.assertEqual([1, 2], self.results()[0]['episodes'])

    def test_sigint_is_boundary_stop_request_not_mid_update_exception(self):
        previous = signal.getsignal(signal.SIGINT)
        def operation(c, path, monitor, **kw):
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)
            self.assertTrue(monitor.interrupt_requested())
            return {'run': str(path), 'complete': True}
        self.fake_runner.run_train = operation
        self.assertEqual(3, self.invoke())
        self.assertEqual('controlled_stop', self.results()[0]['stop'])
        self.assertEqual(previous, signal.getsignal(signal.SIGINT))

    def test_validation_subprocess_does_not_import_heavy_dependencies(self):
        # 真实wrapper的validate路径；检测模块，不靠单个函数import冒充入口。
        script = str(Path(cli.__file__))
        code = ('import runpy,sys; sys.argv=[' + repr(script) + ", 'validate']; "
                'result=runpy.run_path(' + repr(script) + "); result['main'](); "
                "assert not ({'torch','numpy','vizdoom','flyrl.brain'} & set(sys.modules))")
        proc = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                              capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)
        self.assertIn('malecns-short-turn-learning-v1', proc.stdout)


if __name__ == '__main__':
    unittest.main()
