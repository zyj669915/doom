"""下一版编码隔离测试：微型全边图，不安装依赖，不执行训练。"""
from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
_CACHE = tempfile.TemporaryDirectory()
os.environ.setdefault("NUMBA_CACHE_DIR", _CACHE.name)
import numpy as np
import pandas as pd
import torch
from flyrl_next.encoding import Pipeline, _Frontend, _gray84, _resolve, ROOT


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.ids = np.array([2**53 + 1, 7, 9, 23], np.int64)
        self.graph = root / "graph.npz"
        np.savez(self.graph, ptr=np.array([0, 2, 3, 4, 5], np.int64),
            post=np.array([1, 2, 2, 3, 0], np.int32),
            weight=np.array([8, 2, -2, 4, 1], np.float32), ids=self.ids,
            retina=np.array([0, 2], np.int32), uv=np.array([[0, 0], [1, 1]], np.float32),
            lamina=np.array([1], np.int32), sugar=np.array([], np.int32),
            superclass=np.array(["ol_sensory"] * 4))
        paths = {}
        tables = {
            "neurons": pd.DataFrame({"source_id": self.ids[::-1], "cell_type": ["LC1", "T4a", "L2", "L1"]}),
            "catalog": pd.DataFrame({"source_id": self.ids, "superclass": ["ol_sensory"] * 4}),
            "annotations": pd.DataFrame({"bodyId": self.ids[::-1], "somaSide": ["R", "L", "R", "L"], "rootSide": ["R", "L", "R", "L"]}),
        }
        for key, table in tables.items():
            paths[key] = str(root / f"{key}.feather")
            table.to_feather(paths[key])
        self.config = {"graph": str(self.graph), **paths}
        self.black = np.zeros((12, 17, 3), np.uint8)
        self.white = np.full_like(self.black, 255)

    def pipe(self, route, **kwargs):
        return Pipeline({**self.config, "route": route, **kwargs})

    def test_specs_all_routes_and_full_graph(self):
        for route, dim, shape in [("A1", 1968, None), ("A2", 1536, None),
                ("A3", 1536, None), ("A4", 0, (12, 84, 84)),
                ("A5", 0, (4, 84, 84)), ("A6", 0, (4, 84, 84)),
                ("A7", 1536, (4, 84, 84))]:
            with self.subTest(route=route):
                pipe = self.pipe(route)
                self.assertEqual(pipe.spec, {"vector_dim": dim, "image_shape": shape})
                pipe.tic(self.white)
                obs = pipe.observation()
                if dim:
                    self.assertEqual(obs["vector"].shape, (dim,))
                    self.assertEqual(obs["vector"].dtype, np.float32)
                    self.assertEqual(pipe.brain.n, 4)
                    self.assertEqual(len(pipe.brain.post), 5)
                    self.assertEqual(pipe.last_diagnostic["spikes"], int(pipe.brain.counts.sum()))
                    self.assertEqual(pipe.last_diagnostic["spiking_neurons"], int(np.count_nonzero(pipe.brain.counts)))
                    self.assertEqual(pipe.last_diagnostic["integration_set"], int(pipe.brain.nactive[0]))
                    self.assertAlmostEqual(pipe.last_diagnostic["brain_ms"], 28.5)
                else:
                    self.assertNotIn("vector", obs)
                if shape:
                    self.assertEqual(obs["image"].shape, shape)
                    self.assertEqual(obs["image"].dtype, np.uint8)
                else:
                    self.assertNotIn("image", obs)

    def test_pure_visual_does_not_read_graph_annotations(self):
        with patch("numpy.load", side_effect=AssertionError("禁止读图")), patch("pandas.read_feather", side_effect=AssertionError("禁止读注释")):
            for route in ("A4", "A5", "A6"):
                pipe = Pipeline({"route": route, "graph": "missing", "annotations": "missing"})
                pipe.tic(self.white)
                self.assertIsNone(pipe.brain)
                self.assertEqual(pipe.last_diagnostic["brain_ms"], 0)
                pipe.freeze_calibration()
                self.assertTrue(pipe.calibrated)

    def test_images_tic_history_on_off_and_no_crop(self):
        pipe = self.pipe("A4")
        pipe.tic(self.black)
        pipe.tic(self.white)
        image = pipe.observation()["image"]
        np.testing.assert_array_equal(image[-3], 255)
        np.testing.assert_array_equal(image[-2], 255)
        np.testing.assert_array_equal(image[-1], 0)
        pipe.tic(self.black)
        image = pipe.observation()["image"]
        np.testing.assert_array_equal(image[-2], 0)
        np.testing.assert_array_equal(image[-1], 255)
        rgb = self.black.copy()
        rgb[0, 0] = 255
        rgb[-1, -1] = 255
        resized = _gray84(rgb)
        self.assertEqual(resized[0, 0], 255)
        self.assertEqual(resized[-1, -1], 255)
        image.fill(99)
        self.assertNotEqual(pipe.observation()["image"][0, 0, 0], 99)

    def test_a1_decision_history_and_id_partition(self):
        pipe = self.pipe("A1")
        self.assertEqual(len(pipe.groups.group_names), 123)
        np.testing.assert_array_equal(np.sort(np.concatenate(pipe.groups.group_indices)), np.arange(4))
        pipe.tic(self.white)
        first = pipe.raw_vector()
        np.testing.assert_array_equal(first[3*246:4*246], first[7*246:8*246])
        np.testing.assert_array_equal(first, pipe.raw_vector())
        cursor = pipe.brain.cursor
        pipe.observation()
        self.assertEqual(cursor, pipe.brain.cursor)
        # 多个tic只在下一决策时推进一次向量历史。
        pipe.tic(self.black)
        pipe.tic(self.white)
        second = pipe.raw_vector()
        np.testing.assert_array_equal(second[2*246:3*246], first[3*246:4*246])
        np.testing.assert_allclose(second[7*246:], second[3*246:4*246] - first[3*246:4*246])

    def test_hash_stable_exact_ids_and_fast_slow(self):
        a = self.pipe("A2")
        b = self.pipe("A3")
        np.testing.assert_array_equal(a.hash_bucket, b.hash_bucket)
        np.testing.assert_array_equal(a.hash_sign, b.hash_sign)
        different = self.pipe("A2", projection_seed=678)
        self.assertFalse(np.array_equal(a.hash_bucket, different.hash_bucket))
        for rgb in (self.white, self.black, self.white):
            a.tic(rgb)
            b.tic(rgb)
        value = a.raw_vector()
        np.testing.assert_allclose(value[:1024], np.arcsinh(a._hash_rates).ravel(), atol=3e-7)
        self.assertEqual(a.state_dict()["feature_version"], "flyrl-next-encoding-v3-local-contrast-asinh")
        np.testing.assert_array_equal(value, b.raw_vector())
        np.testing.assert_allclose(value[1024:], value[:512] - value[512:1024], atol=5e-7)
        counts = a.brain.counts
        self.assertEqual(counts.shape, a.hash_bucket.shape)

    def test_calibration_welford_freeze_reset_and_weights_only_roundtrip(self):
        pipe = self.pipe("A2")
        one = np.arange(1536, dtype=np.float32) / 200
        two = one + .5
        pipe.calibration_update(one)
        pipe.reset()
        self.assertEqual(pipe.calibration_count, 1)
        pipe.calibration_update(two)
        pipe.freeze_calibration()
        np.testing.assert_allclose(pipe.mu, (one + two) / 2)
        np.testing.assert_allclose(pipe.sd, .25, atol=3e-7)
        with self.assertRaises(RuntimeError):
            pipe.calibration_update(one)
        saved_mean = pipe.mu.copy()
        pipe.reset()
        np.testing.assert_array_equal(pipe.mu, saved_mean)
        memory = BytesIO()
        torch.save(pipe.state_dict(), memory)
        memory.seek(0)
        state = torch.load(memory, weights_only=True)
        other = self.pipe("A2")
        other.load_state_dict(state)
        self.assertTrue(other.calibrated)
        np.testing.assert_array_equal(other.mu, pipe.mu)
        np.testing.assert_array_equal(other.hash_bucket, pipe.hash_bucket)
        self.assertEqual(other.brain.cursor, 0)
        for rgb in (self.white, self.black):
            pipe.tic(rgb)
            other.tic(rgb)
        np.testing.assert_array_equal(pipe.observation()["vector"], other.observation()["vector"])
        other.start_calibration()
        self.assertFalse(other.calibrated)
        self.assertEqual(other.calibration_count, 0)

    def test_default_online_calibration_and_actual_35hz_time(self):
        pipe = self.pipe("A2")
        samples = []
        for _ in range(7):
            pipe.tic(self.white)
            samples.append(pipe.raw_vector())
            pipe.calibration_update()
        pipe.freeze_calibration()
        np.testing.assert_allclose(pipe.mu, np.mean(np.asarray(samples, np.float64), axis=0), atol=3e-7)
        self.assertAlmostEqual(pipe.brain.sim_ms, 200.)
        self.assertEqual(pipe.calibration_count, 7)

    def test_checkpoint_rejects_bad_semantics_hash_and_statistics(self):
        pipe = self.pipe("A2")
        state = pipe.state_dict()
        for key, value in [("graph_signature", "bad"), ("schema", 9),
                           ("hash_bucket", torch.ones_like(state["hash_bucket"]) * 511),
                           ("sd", torch.zeros(1536)), ("mu", np.zeros(1536)),
                           ("calibration_count", -1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                pipe.load_state_dict({**state, key: value})
        with self.assertRaises(ValueError):
            self.pipe("A3").load_state_dict(state)

    def test_frontends_real_inputs_and_threshold_bias_scale(self):
        uv = np.array([[0, 0], [1, 1]], np.float32)
        rgb = self.white.copy()
        rgb[0, 0] = 80
        raw = _Frontend(uv, "raw", "none", 0)
        adapt = _Frontend(uv, "adapt", "none", 0)
        contrast = _Frontend(uv, "contrast", "none", 0)
        for frontend in (raw, adapt):
            frontend.step(self.black, 1000/35)
        lum = raw.step(rgb, 1000/35)
        self.assertFalse(np.allclose(lum, adapt.step(rgb, 1000/35)))
        self.assertFalse(np.allclose(lum, contrast.step(rgb, 1000/35)))
        for frontend in ("raw", "adapt", "contrast", "agc_motion"):
            pipe = self.pipe("A3", frontend=frontend, scale=1., bias=12., threshold=-45.)
            pipe.tic(rgb)
            self.assertEqual(pipe.brain.threshold, -45.)
            self.assertEqual(pipe.brain.bias, 12.)
            self.assertEqual(pipe.brain.scale, 1.)
            np.testing.assert_array_equal(pipe.brain.weight, np.array([8, 2, -2, 4, 1], np.float32))
        black = _Frontend(uv, "adapt", "black", 0)
        np.testing.assert_array_equal(black.step(self.white, 30), [0, 0])

    def test_contrast_uses_local_spatial_neighborhood_and_vision_sampling(self):
        from flyrl.encoding import Vision
        from flyrl_next.encoding import _local_contrast_rgb
        # 两受体处原像素相同，但周围亮暗不同；全局受体标准化无法区分。
        rgb = np.full((25, 45, 3), 30, np.uint8)
        rgb[:, 23:] = 220
        rgb[12, 10] = rgb[12, 34] = 128
        uv = np.array([[10/44, .5], [34/44, .5]], np.float64)
        raw = _Frontend(uv, "raw", "none", 0).step(rgb, 30)
        np.testing.assert_allclose(raw[0], raw[1])
        contrast = _Frontend(uv, "contrast", "none", 0).step(rgb, 30)
        self.assertGreater(contrast[0], contrast[1] + .3)
        expected = Vision(uv, "raw").step(_local_contrast_rgb(rgb))
        np.testing.assert_array_equal(contrast, expected)
        # 远处变化不应影响固定9x9局部邻域。
        remote = rgb.copy()
        remote[0:3, :, :] = 255
        np.testing.assert_array_equal(contrast, _Frontend(uv, "contrast", "none", 0).step(remote, 30))
        uniform = np.full_like(rgb, 180)
        response = _Frontend(uv, "contrast", "none", 0).step(uniform, 30)
        np.testing.assert_allclose(response, .5, atol=.004)

    def test_input_validation_paths_and_pickle_refusal(self):
        self.assertEqual(_resolve("abc"), ROOT / "abc")
        pipe = self.pipe("A5")
        for bad in (np.zeros((3, 4)), np.zeros((3, 4, 3)), np.zeros((0, 4, 3), np.uint8)):
            with self.assertRaises(ValueError):
                pipe.tic(bad)
        unsafe = Path(self.tmp.name) / "unsafe.npz"
        np.savez(unsafe, ptr=np.array([object()], dtype=object))
        with self.assertRaises(ValueError):
            Pipeline({"route": "A2", "graph": str(unsafe)})
        with self.assertRaises(ValueError):
            Pipeline({"route": "A8"})


if __name__ == "__main__":
    unittest.main()
