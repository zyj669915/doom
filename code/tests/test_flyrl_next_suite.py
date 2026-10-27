"""仅测试协议和调度门禁；绝不启动真实训练。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flyrl_next import evaluation, suite


def result():
    return {"config": {"task": "corridor", "route": "A1", "seed": 1, "rewire_seed": 2,
                       "action_set": "full", "action_repeat": 4, "max_turn": 6,
                       "objective": "potential", "rewards": {}, "max_game_tics": 100,
                       "wall_hours": 2, "calibration_steps": 2},
            "split": "test", "quick": False, "frozen": True,
            "intervention": "none", "policy": "learned", "training_game_tics": 100,
            "checkpoint": "checkpoint-1", "checkpoint_sha256": "hash",
            "provenance": {"sources": {"runner": "a"}, "inputs": {"iwad": "b"}, "versions": {"python": "3"}},
            "episodes": [{"seed": 3, "layout": "test", "assets": {"map": "hash"},
                          "return": 1., "tics": 20, "success": True, "max_progress": 1.}]}


class CompareTests(unittest.TestCase):
    def test_corridor_ranking_success_then_progress_then_success_tics(self):
        a, b = result(), result()
        a["checkpoint"], b["checkpoint"] = "a", "b"
        a["episodes"][0].update(success=False, max_progress=.9, tics=1, **{"return": 100})
        b["episodes"][0].update(success=True, max_progress=.8, tics=100, **{"return": -100})
        self.assertEqual(evaluation.compare([a, b])["leaderboards"][0]["runs"][0]["checkpoint"], "b")
        a["episodes"][0].update(success=True, max_progress=.9)
        self.assertEqual(evaluation.compare([a, b])["leaderboards"][0]["runs"][0]["checkpoint"], "a")
        b["episodes"][0].update(max_progress=.9, tics=.5)
        self.assertEqual(evaluation.compare([a, b])["leaderboards"][0]["runs"][0]["checkpoint"], "b")
        a["episodes"][0].update(success=False)
        b["episodes"][0].update(success=False)
        board = evaluation.compare([a, b])["leaderboards"][0]
        self.assertEqual([r["checkpoint"] for r in board["runs"]], ["a", "b"])
        self.assertIsNone(board["runs"][0]["summary"]["successful_tics_mean"])

    def test_performance_action_separation(self):
        a, b = result(), result()
        b["config"]["action_set"] = "forward_only"
        report = evaluation.compare([a, b])
        self.assertEqual(len(report["leaderboards"]), 2)
        self.assertFalse(report["cross_action_comparable"])
        with self.assertRaises(ValueError):
            evaluation.compare([a, b], "intervention")

    def test_performance_allows_route_but_records_actual_budget(self):
        a, b = result(), result()
        b["config"].update(route="A4", algorithm="ppo", frontend="raw")
        a["config"].update(algorithm="dqn", frontend="agc_motion")
        b["training_game_tics"] = 80
        board = evaluation.compare([a, b])["leaderboards"][0]
        self.assertFalse(board["actual_training_budget_equal"])

    def test_strict_protocol(self):
        for mutate in (lambda x: x["config"].update(objective="progress_delta"),
                       lambda x: x.update(split="val"),
                       lambda x: x["config"].update(max_game_tics=200),
                       lambda x: x["episodes"][0].update(seed=99),
                       lambda x: x["provenance"]["sources"].update(runner="changed"),
                       lambda x: x.update(quick=True)):
            a, b = result(), result()
            mutate(b)
            with self.assertRaises(ValueError):
                evaluation.compare([a, b])

    def test_intervention_same_checkpoint(self):
        a, b = result(), result()
        b["intervention"] = "black"
        evaluation.compare([a, b], "intervention")
        b["checkpoint_sha256"] = "other"
        with self.assertRaises(ValueError):
            evaluation.compare([a, b], "intervention")

    def test_connectome_only_graph_manipulation(self):
        a, b = result(), result()
        a["config"]["arm"] = "real"
        b["config"].update(arm="rewired", rewire_seed=9)
        evaluation.compare([a, b], "connectome")
        b["config"]["seed"] = 99
        with self.assertRaises(ValueError):
            evaluation.compare([a, b], "connectome")

    def test_connectome_rejects_encoding_routes_without_topology_arm(self):
        a, b = result(), result()
        b["config"]["route"] = "A2"
        with self.assertRaisesRegex(ValueError, "A1/A2"):
            evaluation.compare([a, b], "connectome")

    def test_shared_brain_input_hash_cannot_change(self):
        a, b = result(), result()
        a["provenance"]["inputs"]["graph"] = {"sha256": "a"}
        b["provenance"]["inputs"]["graph"] = {"sha256": "b"}
        with self.assertRaises(ValueError):
            evaluation.compare([a, b])
        del b["provenance"]["inputs"]["graph"]
        evaluation.compare([a, b])

    def test_combat_metrics_are_observed_not_fabricated(self):
        a = result()
        a["config"]["task"] = "combat"
        a["episodes"][0]["telemetry"] = {"kills": 2, "hits": 3, "health": 40}
        report = evaluation.compare([a, deepcopy(a)])
        summary = report["leaderboards"][0]["runs"][0]["summary"]
        self.assertEqual(summary["kills_mean"], 2)
        self.assertEqual(summary["terminal_health_mean"], 40)
        self.assertNotIn("success_rate", summary)
        del a["episodes"][0]["telemetry"]["hits"]
        with self.assertRaises(ValueError):
            evaluation.compare([a, a])


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        jobs = []
        for name, route in (("brain1", "A1"), ("brain2", "A2"), ("cnn", "A4")):
            config = self.root / (name + ".json")
            config.write_text(json.dumps({"route": route, "device": "cpu"}), encoding="utf-8")
            jobs.append({"id": name, "config": config.name,
                         "resources": {"ram_gib": 1, "commit_gib": 2, "vram_gib": 0}})
        self.path = self.root / "suite.json"
        self.path.write_text(json.dumps({"runs": jobs, "output": "out"}), encoding="utf-8")
        self.available = {"ram_available": 10 * suite.GIB, "commit_available": 10 * suite.GIB, "vram_available": None}

    def test_default_serial_explicit_one_brain_one_cnn(self):
        with patch.object(suite, "probe_resources", return_value=self.available):
            default = suite.plan(self.path)
            parallel = suite.plan(self.path, parallel=True)
        self.assertTrue(all(len(b["ids"]) == 1 for b in default["batches"]))
        self.assertEqual(parallel["batches"][0]["ids"], ["brain1", "cnn"])
        self.assertEqual(parallel["batches"][1]["ids"], ["brain2"])
        self.assertIn("flyrl_next_cli.py", default["jobs"][0]["command"][1])

    def test_ram_reserve_commit_unknown_fail_closed(self):
        for resources in ({**self.available, "ram_available": 3 * suite.GIB},
                          {**self.available, "commit_available": 0},
                          {**self.available, "commit_available": None},
                          {**self.available, "ram_available": None}):
            with patch.object(suite, "probe_resources", return_value=resources), patch.object(suite.subprocess, "Popen") as popen:
                self.assertFalse(suite.plan(self.path)["allowed"])
                with self.assertRaises(RuntimeError):
                    suite.run(self.path)
                popen.assert_not_called()

    def test_gpu_unknown_rejected_when_required(self):
        job = {"requirements": {"ram_gib": suite.GIB, "commit_gib": suite.GIB, "vram_gib": suite.GIB}}
        self.assertTrue(suite._gate([job], self.available))

    def test_stop_is_file_only_and_status_readonly(self):
        folder = self.root / "out"
        folder.mkdir()
        with patch.object(suite.subprocess, "Popen") as popen:
            report = suite.stop(folder)
            self.assertTrue((folder / "STOP").exists())
            self.assertFalse(report["processes_terminated"])
            self.assertTrue(suite.status(folder)["stop_requested"])
            popen.assert_not_called()

    def test_run_rechecks_resources_without_launch(self):
        depleted = {**self.available, "ram_available": 0}
        with patch.object(suite, "probe_resources", side_effect=[self.available, depleted]), patch.object(suite.subprocess, "Popen") as popen:
            with self.assertRaises(RuntimeError):
                suite.run(self.path)
            popen.assert_not_called()
        self.assertFalse((self.root / "out" / ".suite.lock").exists())

    def test_current_process_memory_scope_and_independent_json(self):
        output = self.root / "memory-observation.json"
        original = self.path.read_bytes()
        with patch.object(suite.subprocess, "Popen") as popen:
            report = suite.current_process_memory(output)
            popen.assert_not_called()
        self.assertEqual(report["scope"], "calling_process_lifetime_not_children")
        self.assertIsNone(report["resources"])
        self.assertIsNone(report["vram_bytes"])
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
        self.assertEqual(original, self.path.read_bytes())
        if report["available"]:
            self.assertGreater(report["peak_rss_bytes"], 0)
            self.assertGreaterEqual(report["peak_rss_bytes"], report["rss_bytes"])
            self.assertGreaterEqual(report["peak_commit_bytes"], report["private_commit_bytes"])

    def test_memory_helper_unknown_platform_fail_closed(self):
        with patch.object(suite.sys, "platform", "unsupported"):
            report = suite.current_process_memory()
        self.assertFalse(report["available"])
        self.assertIsNone(report["peak_commit_bytes"])
        self.assertIsNone(report["resources"])

    def test_run_uses_subprocess_not_train_import(self):
        class Process:
            pid = 123
            def wait(self):
                return 0
        with patch.object(suite, "probe_resources", return_value=self.available), patch.object(suite.subprocess, "Popen", return_value=Process()) as popen:
            report = suite.run(self.path)
        self.assertEqual(len(report["runs"]), 3)
        self.assertEqual(popen.call_count, 3)
        command = popen.call_args.args[0]
        self.assertEqual(command[2:4], ["train", "--config"])
        self.assertTrue(Path(command[-1]).name == "effective-config.json")


if __name__ == "__main__":
    unittest.main()
