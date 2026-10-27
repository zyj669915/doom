"""小图/纯数组测试；可直接用 python -I 运行，不需要 pytest 或全图仿真。"""
import ast
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

# -I 不自动添加项目路径；只显式添加本项目代码，不导入实验入口脚本。
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
_CACHE = tempfile.TemporaryDirectory()
os.environ["NUMBA_CACHE_DIR"] = _CACHE.name
import numpy as np
import pandas as pd
from flyrl import FrozenBrain, MultiscaleRates, Vision, build_groups, rewire_posts
from flyrl.brain import advance_th


def write_graph(path):
    """五节点带正负边、自环、空出度节点的小图。"""
    np.savez(path, ptr=np.array([0, 2, 4, 6, 7, 7], np.int64),
             post=np.array([1, 2, 0, 3, 1, 4, 3], np.int32),
             weight=np.array([8, 4, -5, -2, 3, 6, 1], np.float32),
             ids=np.array([91, 20, 73, 44, 55], np.int64),
             retina=np.array([0], np.int32), uv=np.array([[.5, .5]], np.float32),
             lamina=np.array([1], np.int32), sugar=np.array([2], np.int32),
             superclass=np.array(["ol_sensory"] * 5))


def groups_fixture():
    # 故意使用大于 2**53 的 ID 和不同表顺序，防止浮点转换或行序对齐。
    ids = np.array([2**53 + 1, 73, 91, 44], np.int64)
    nodes = pd.DataFrame({"source_id": [91, 44, 2**53 + 1, 73],
                          "cell_type": ["DNp20", "T4a", "L1", "mystery"],
                          "side": ["R", "left", "L", None],
                          "superclass": ["wrong"] * 4})
    catalog = pd.DataFrame({"source_id": [73, 2**53 + 1, 44, 91],
                            "superclass": ["cb_intrinsic", "ol_intrinsic", "ol_intrinsic", "descending_neuron"]})
    return build_groups(ids, nodes, catalog), nodes, catalog


class EncodingTests(unittest.TestCase):
    def test_group_id_alignment_partition_and_save(self):
        groups, nodes, catalog = groups_fixture()
        self.assertEqual(len(groups.group_names), 123)
        lookup = dict(zip(groups.group_names, groups.group_indices))
        np.testing.assert_array_equal(lookup["cell_type:L1:L"], [0])
        np.testing.assert_array_equal(lookup["cell_type:DNp20:R"], [2])
        np.testing.assert_array_equal(lookup["cell_type:T4:L"], [3])
        np.testing.assert_array_equal(lookup["superclass_residual:cb_intrinsic:unknown"], [1])
        self.assertEqual(len(lookup["superclass_residual:ol_intrinsic:L"]), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate(groups.group_indices)), np.arange(4))
        shuffled = build_groups(groups.graph_ids, nodes.iloc[::-1], catalog.iloc[::-1])
        for a, b in zip(groups.group_indices, shuffled.group_indices):
            np.testing.assert_array_equal(a, b)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "groups.npz"
            groups.save(path)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["graph_ids"], groups.graph_ids)
                self.assertEqual(saved["offsets"][-1], 4)
                self.assertEqual(len(saved["annotations"]), 123)

    def test_group_missing_duplicate_float_and_side_override(self):
        groups, nodes, catalog = groups_fixture()
        for ids in ([1], [73, 73], [73.0]):
            with self.assertRaises(ValueError):
                build_groups(ids, nodes, catalog)
        with self.assertRaises(ValueError):
            build_groups(groups.graph_ids, pd.concat([nodes, nodes.iloc[:1]]), catalog)
        result = build_groups(groups.graph_ids, nodes, catalog, side_by_id={73: "right"})
        at = result.group_names.index("superclass_residual:cb_intrinsic:R")
        np.testing.assert_array_equal(result.group_indices[at], [1])

    def test_population_mean_and_feather_input(self):
        nodes = pd.DataFrame({"source_id": [3, 1, 2], "cell_type": ["L1", "L1", "other"],
                              "side": ["L", "L", "R"]})
        catalog = pd.DataFrame({"source_id": [2, 3, 1],
                                "superclass": ["unknown", "ol_intrinsic", "ol_intrinsic"]})
        with tempfile.TemporaryDirectory() as tmp:
            node_path, cat_path = Path(tmp) / "neurons.feather", Path(tmp) / "catalog.feather"
            nodes.to_feather(node_path)
            catalog.to_feather(cat_path)
            groups = build_groups([1, 2, 3], node_path, cat_path)
        rates = MultiscaleRates(groups)
        feature = rates.step([1, 0, 3], 20).reshape(2, -1)
        at = groups.group_names.index("cell_type:L1:L")
        np.testing.assert_allclose(feature[:, at],
            np.log1p(100 * (1 - np.exp(-20 / np.array([50., 200.])))), rtol=1e-6)
        other = groups.group_names.index("superclass_residual:other:R")
        np.testing.assert_array_equal(groups.group_indices[other], [1])

    def test_actual_time_multiscale_and_reset(self):
        groups, _, _ = groups_fixture()
        rates = MultiscaleRates(groups)
        at = groups.group_names.index("cell_type:L1:L")
        counts = np.array([2, 0, 0, 0])
        feat = rates.step(counts, 20)
        expected = 100 * (1 - np.exp(-20 / np.array([50., 200.])))
        np.testing.assert_allclose(feat.reshape(2, -1)[:, at], np.log1p(expected), rtol=1e-6)
        # 恒定 Hz 下，两个 20ms 应与一个 40ms 逐元素一致，而不是按帧滤波。
        two = rates.step(counts, 20)
        rates.reset()
        one = rates.step(counts * 2, 40)
        np.testing.assert_allclose(two, one, rtol=1e-6)
        self.assertEqual(rates.dim, 246)
        rates.reset()
        self.assertEqual(rates.elapsed_ms, 0)
        self.assertFalse(rates.rates.any())
        np.testing.assert_array_equal(rates.step(np.zeros(4), 0), np.zeros(246))
        for duration in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                rates.step(counts, duration)
        with self.assertRaises(ValueError):
            rates.step(counts, 0)

    def test_raw_and_agc_black_reset_scramble(self):
        uv = np.array([[0, 0], [1, 1], [.5, .5], [1, 0]])
        black = np.zeros((2, 2, 3), np.uint8)
        white = np.full_like(black, 255)
        image = black.copy()
        image[0, 0] = 255
        for frontend in ("raw", "agc_motion"):
            vision = Vision(uv, frontend=frontend)
            np.testing.assert_array_equal(vision(black), np.zeros(4))
            vision.reset()
            first = vision(image)
            vision(white)
            np.testing.assert_array_equal(vision(white, "black"), np.zeros(4))
            vision.reset()
            np.testing.assert_array_equal(vision(image), first)
            scrambled = Vision(uv, frontend=frontend, intervention="scramble", seed=123)
            plain = Vision(uv, frontend=frontend)
            np.testing.assert_allclose(np.sort(scrambled(image)), np.sort(plain(image)))
            perm = scrambled.permutation.copy()
            scrambled.reset()
            np.testing.assert_array_equal(scrambled.permutation, perm)
        np.testing.assert_allclose(Vision(uv)(image), [1, 0, .25, 0], atol=1e-6)
        with self.assertRaises(ValueError):
            Vision(uv)(black.astype(float))

    def test_agc_motion_formula(self):
        uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        image = np.zeros((2, 2, 3), np.uint8)
        image[0, 0] = 180
        image2 = image[::-1].copy()
        raw = Vision(uv)
        first, second = raw(image), raw(image2)
        vision = Vision(uv, frontend="agc_motion")
        vision(image)
        motion = np.clip(second + 1.2 * (second - first), 0, 1)
        expected = .5 + .5 * np.tanh((motion-motion.mean()) / (motion.std()+1e-4) / 1.5)
        np.testing.assert_allclose(vision(image2), expected, rtol=1e-6)


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "graph.npz"
        write_graph(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_time_remainder_35hz_and_substep(self):
        brain = FrozenBrain(self.path)
        actual = [brain.step([0], 1000 / 35).actual_ms for _ in range(35)]
        self.assertEqual(brain.cursor, 10000)
        self.assertAlmostEqual(sum(actual), 1000)
        self.assertLess(brain.remainder_ms, 1e-8)
        self.assertTrue({round(ms, 1) for ms in actual} == {28.5, 28.6})
        brain.reset()
        self.assertEqual(brain.step([0], .04).actual_ms, 0)
        self.assertEqual(brain.step([0], .06).actual_ms, .1)
        brain.reset()
        rng = np.random.default_rng(7)
        total = 0
        for duration in rng.uniform(.01, .7, 100):
            total += duration
            brain.step([0], duration)
            self.assertLessEqual(brain.sim_ms, total + 1e-8)
            self.assertLess(total - brain.sim_ms, .1 + 1e-8)

    def test_reset_all_states_and_frozen_weights(self):
        brain = FrozenBrain(self.path)
        weights = brain.weight.copy()
        first = brain.step([1], 37.57)
        brain.reset()
        self.assertEqual(brain.cursor, 0)
        self.assertEqual(brain.remainder_ms, 0)
        self.assertEqual(brain.total_spikes, 0)
        np.testing.assert_array_equal(brain.v, np.full(5, -52))
        for key in ("g", "drive", "refractory", "queue", "queue_count", "counts", "luminance"):
            self.assertFalse(getattr(brain, key).any(), key)
        np.testing.assert_array_equal(brain.active_flag, [1, 1, 1, 0, 0])
        self.assertEqual(brain.nactive[0], 3)
        second = brain.step([1], 37.57)
        np.testing.assert_array_equal(first.counts, second.counts)
        np.testing.assert_array_equal(weights, brain.weight)
        self.assertFalse(brain.weight.flags.writeable)
        brain.reset(warmup_ms=1.25)
        self.assertAlmostEqual(brain.sim_ms, 1.2)
        self.assertAlmostEqual(brain.remainder_ms, .05)

    def test_baseline_reference_parity_and_custom_threshold(self):
        # 只解析 doom.engine 的已审核函数为纯 Python reference，不导入/执行上游模块。
        engine = CODE.parents[1] / "doomfly" / "doom" / "engine.py"
        tree = ast.parse(engine.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "advance")
        function.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {"math": math}
        exec(compile(module, str(engine), "exec"), namespace)
        brain = FrozenBrain(self.path, scale=1, bias=12, threshold=-45)
        reference = FrozenBrain(self.path, scale=1, bias=12, threshold=-45)
        for lum in ([1], [.2], [0], [.8]):
            result = brain.step(lum, 17.3)
            steps = int(round(result.actual_ms / .1))
            alpha = 1 - math.exp(-result.actual_ms / 10)
            reference.luminance += alpha * (np.clip(lum, 0, 1) - reference.luminance)
            reference.drive.fill(0)
            reference.drive[reference.lamina] = 12
            reference.drive[reference.retina] = 30 * reference.luminance / (.02 + reference.luminance)
            reference.counts.fill(0)
            reference.cursor = namespace["advance"](reference.ptr, reference.post, reference.weight,
                reference.v, reference.g, reference.refractory, reference.drive, reference.queue,
                reference.queue_count, reference.cursor, steps, .1, reference.counts,
                reference.active, reference.active_flag, reference.nactive)
            for key in ("counts", "refractory", "queue", "queue_count", "active", "active_flag", "nactive"):
                np.testing.assert_array_equal(getattr(brain, key), getattr(reference, key), err_msg=key)
            np.testing.assert_allclose(brain.v, reference.v, rtol=1e-5, atol=2e-5)
            np.testing.assert_allclose(brain.g, reference.g, rtol=1e-5, atol=2e-5)
        low = FrozenBrain(self.path, scale=1, bias=6, threshold=-48)
        high = FrozenBrain(self.path, scale=1, bias=6, threshold=-45)
        self.assertGreater(low.step([0], 50).counts.sum(), high.step([0], 50).counts.sum())

    def test_tuned_kernel_original_reference_parity(self):
        # 只提取函数 AST，不执行 readout_experiment 顶层的目录创建或仿真逻辑。
        source_path = CODE / "readout_experiment.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "advance_th")
        function.decorator_list = []
        namespace = {"math": math}
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, str(source_path), "exec"), namespace)
        actual, reference = FrozenBrain(self.path), FrozenBrain(self.path)
        for duration, lum in ((13.7, [1]), (21.2, [.1]), (8.6, [0])):
            result = actual.step(lum, duration)
            alpha = -math.expm1(-result.actual_ms / 10)
            reference.luminance += alpha * (np.clip(lum, 0, 1) - reference.luminance)
            reference.drive.fill(0)
            reference.drive[reference.lamina] = 6
            reference.drive[reference.retina] = 30 * reference.luminance / (.02 + reference.luminance)
            reference.counts.fill(0)
            reference.cursor = namespace["advance_th"](reference.ptr, reference.post, reference.weight,
                reference.v, reference.g, reference.refractory, reference.drive, reference.queue,
                reference.queue_count, reference.cursor, int(round(result.actual_ms / .1)), .1,
                reference.counts, reference.active, reference.active_flag, reference.nactive, -48)
            np.testing.assert_array_equal(actual.counts, reference.counts)
            np.testing.assert_array_equal(actual.queue_count, reference.queue_count)
            np.testing.assert_allclose(actual.v, reference.v, rtol=1e-5, atol=2e-5)
            np.testing.assert_allclose(actual.g, reference.g, rtol=1e-5, atol=2e-5)

    def test_rewire_degrees_sign_degrees_and_disk_unchanged(self):
        original = self.path.read_bytes()
        brain = FrozenBrain(self.path)
        ptr = brain.ptr
        post, report = rewire_posts(ptr, brain.post, brain.weight, 12)
        again, _ = rewire_posts(ptr, brain.post, brain.weight, 12)
        np.testing.assert_array_equal(post, again)
        np.testing.assert_array_equal(np.bincount(post, minlength=5), np.bincount(brain.post, minlength=5))
        for sign in (-1, 0, 1):
            mask = np.sign(brain.weight) == sign
            np.testing.assert_array_equal(np.bincount(post[mask], minlength=5),
                                          np.bincount(brain.post[mask], minlength=5))
        self.assertEqual(len(post), ptr[-1])
        self.assertFalse(report["preserves_simple_graph"])
        for label, endpoints in (("before", brain.post), ("after", post)):
            source = np.repeat(np.arange(5), np.diff(ptr))
            pairs = list(zip(source.tolist(), endpoints.tolist()))
            self.assertEqual(report[label]["self_edges"], sum(i == j for i, j in pairs))
            self.assertEqual(report[label]["duplicate_edges"], len(pairs) - len(set(pairs)))
        rewired = FrozenBrain(self.path, rewired=True, rewire_seed=12)
        np.testing.assert_array_equal(rewired.post, post)
        np.testing.assert_array_equal(rewired.weight, brain.weight)
        self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaises(ValueError):
            rewire_posts(np.array([0, 2]), np.array([0, 0]), np.array([-1., 1.]))
        # 混合多个空出度行的 reduceat 路径。
        result, _ = rewire_posts(np.array([0, 0, 2, 2, 3, 3]), np.array([1, 4, 0]), np.array([-1., -2., 3.]))
        np.testing.assert_array_equal(np.sort(result), [0, 1, 4])

    def test_rewire_reports_duplicate_creation(self):
        ptr = np.array([0, 2, 4, 6, 8])
        post = np.array([1, 2, 0, 3, 0, 3, 1, 2])
        weight = np.ones(8)
        reports = [rewire_posts(ptr, post, weight, seed)[1] for seed in range(10)]
        self.assertTrue(any(report["after"]["duplicate_edges"] > 0 for report in reports))
        self.assertTrue(any(report["delta"]["self_edges"] != 0 for report in reports))

    def test_black_diagnostic_detached(self):
        brain = FrozenBrain(self.path)
        brain.step([1], 2.37)
        keys = ("v", "g", "drive", "refractory", "queue", "queue_count", "counts", "luminance", "active", "active_flag", "nactive", "weight")
        before = {key: getattr(brain, key).copy() for key in keys}
        cursor, remainder, sim_ms = brain.cursor, brain.remainder_ms, brain.sim_ms
        result = brain.black_run([3.3, 4.4], warmup_ms=1)
        self.assertFalse(result["weights_trained"])
        self.assertAlmostEqual(result["actual_ms"], 7.7)
        self.assertAlmostEqual(result["warmup_actual_ms"], 1)
        for key in keys:
            np.testing.assert_array_equal(getattr(brain, key), before[key])
        self.assertEqual((brain.cursor, brain.remainder_ms, brain.sim_ms), (cursor, remainder, sim_ms))
        control = FrozenBrain(self.path)
        control.reset(warmup_ms=1)
        expected = control.step([0], 3.3).counts + control.step([0], 4.4).counts
        np.testing.assert_array_equal(result["counts"], expected)

    def test_validation_and_no_side_effect_script_import(self):
        with self.assertRaises(ValueError):
            FrozenBrain(self.path, dt=.2)
        brain = FrozenBrain(self.path)
        for duration in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                brain.step([0], duration)
        with self.assertRaises(ValueError):
            brain.step([0, 1], 1)
        # 在独立解释器检查模块自己的导入副作用，不误判其他测试已导入 torch。
        import subprocess
        code = ("import sys;sys.path.insert(0," + repr(str(Path(__file__).resolve().parents[1])) + ");"
                "import flyrl.brain,flyrl.encoding;"
                "assert not {'readout_experiment','torch','vizdoom'} & set(sys.modules)")
        subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True, capture_output=True)
        with np.load(self.path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["post"] = np.array([99] * 7, np.int32)
        np.savez(self.path, **arrays)
        with self.assertRaises(ValueError):
            FrozenBrain(self.path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
