"""资源门禁与监视器的隔离测试；只用标准库和模拟数据，不加载连接组。"""

import ctypes
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "flyrl_stage" / "resources.py"
_SPEC = importlib.util.spec_from_file_location("stage_resources_under_test", _MODULE_PATH)
resources = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(resources)


def sample(ram=8 * resources.GIB, commit=9 * resources.GIB, rss=100, private=200):
    """构造完整而独立的系统/进程树快照。"""
    return {
        "ok": True,
        "errors": [],
        "ram_available_bytes": ram,
        "commit_available_bytes": commit,
        "process_tree": {
            "rss_bytes": rss,
            "private_commit_bytes": private,
            "children_snapshot_count": 1,
            "children_snapshot_rss_bytes": 30,
            "children_snapshot_private_commit_bytes": 40,
            "child_pids": [202],
            "main_process": {
                "peak_working_set_bytes": 500,
                "peak_pagefile_usage_bytes": 600,
            },
        },
    }


class GateTests(unittest.TestCase):
    def test_gib_budget_preserves_fixed_reserve_for_both_resources(self):
        with patch.object(resources, "probe_resources", return_value=sample()):
            result = resources.gate(5, 6)
        self.assertTrue(result["allowed"])
        self.assertEqual(result["reserve_ram_bytes"], 3 * resources.GIB)
        self.assertEqual(result["reserve_commit_bytes"], 3 * resources.GIB)

    def test_one_byte_below_ram_or_commit_reserve_is_rejected(self):
        for key in ("ram", "commit"):
            values = {key: 4 * resources.GIB - 1}
            with self.subTest(key=key), patch.object(
                resources, "probe_resources", return_value=sample(**values)
            ):
                self.assertFalse(resources.gate(1, 1)["allowed"])

    def test_byte_budgets_and_fractional_gib_round_up(self):
        with patch.object(resources, "probe_resources", return_value=sample()):
            result = resources.gate(need_ram_bytes=5 * resources.GIB, need_commit_bytes=1)
            fractional = resources.gate(0.1, 0)
        self.assertTrue(result["allowed"])
        self.assertEqual(fractional["need_ram_bytes"], 107374183)

    def test_invalid_missing_or_ambiguous_budget_fails_closed(self):
        cases = [
            {},
            {"ram_gib": -1, "commit_gib": 0},
            {"ram_gib": float("nan"), "commit_gib": 0},
            {"ram_gib": float("inf"), "commit_gib": 0},
            {"ram_gib": True, "commit_gib": 0},
            {"need_ram_bytes": 1.5, "need_commit_bytes": 0},
            {"ram_gib": 0, "need_ram_bytes": 0, "commit_gib": 0},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), patch.object(
                resources, "probe_resources", return_value=sample()
            ):
                self.assertFalse(resources.gate(**kwargs)["allowed"])

    def test_probe_errors_unknown_and_exception_cannot_bypass_gate(self):
        unknown = sample()
        unknown["ram_available_bytes"] = None
        failed = sample()
        failed["ok"] = False
        failed["errors"] = ["进程树采样失败。"]
        for snapshot in (unknown, failed, {}):
            with self.subTest(snapshot=snapshot), patch.object(
                resources, "probe_resources", return_value=snapshot
            ):
                self.assertFalse(resources.gate(0, 0)["allowed"])
        with patch.object(resources, "probe_resources", side_effect=OSError("测试错误")):
            self.assertFalse(resources.gate(0, 0)["allowed"])


class ProbeTests(unittest.TestCase):
    def test_global_memory_status_ex_uses_available_physical_and_commit(self):
        def fill(pointer):
            status = ctypes.cast(pointer, ctypes.POINTER(resources._MemoryStatusEx)).contents
            self.assertEqual(status.dwLength, ctypes.sizeof(resources._MemoryStatusEx))
            status.ullAvailPhys = 1234
            status.ullAvailPageFile = 5678
            status.ullTotalPhys = 10000
            status.ullTotalPageFile = 20000
            return True

        kernel = Mock()
        kernel.GlobalMemoryStatusEx.side_effect = fill
        with patch.object(resources, "_windows_api", return_value=(kernel, Mock())):
            result = resources._system_memory()
        self.assertEqual(result["ram_available_bytes"], 1234)
        self.assertEqual(result["commit_available_bytes"], 5678)
        self.assertEqual(result["commit_limit_bytes"], 20000)

    def test_recursive_descendants_are_summed_once_and_unrelated_pids_excluded(self):
        parents = {10: 1, 20: 10, 30: 20, 40: 1, 50: 30}

        def memory(pid):
            return {
                "pid": pid,
                "rss_bytes": pid,
                "private_commit_bytes": 2 * pid,
                "peak_working_set_bytes": 3 * pid,
                "peak_pagefile_usage_bytes": 4 * pid,
            }

        with patch.object(resources, "_snapshot_parent_pids", return_value=parents), patch.object(
            resources, "_process_memory", side_effect=memory
        ) as read:
            tree = resources._sample_process_tree(10)
        self.assertEqual(tree["child_pids"], [20, 30, 50])
        self.assertEqual(tree["rss_bytes"], 110)
        self.assertEqual(tree["private_commit_bytes"], 220)
        self.assertEqual(tree["children_snapshot_count"], 3)
        self.assertEqual(tree["children_snapshot_rss_bytes"], 100)
        self.assertEqual(tree["main_process"]["peak_pagefile_usage_bytes"], 40)
        self.assertEqual(read.call_count, 4)

    def test_process_memory_counters_and_handle_cleanup(self):
        kernel = Mock()
        kernel.OpenProcess.return_value = 123
        psapi = Mock()

        def fill(handle, pointer, size):
            counters = ctypes.cast(
                pointer, ctypes.POINTER(resources._ProcessMemoryCountersEx)
            ).contents
            self.assertEqual(size, ctypes.sizeof(counters))
            counters.WorkingSetSize = 101
            counters.PrivateUsage = 202
            counters.PeakWorkingSetSize = 303
            counters.PeakPagefileUsage = 404
            return True

        psapi.GetProcessMemoryInfo.side_effect = fill
        with patch.object(resources, "_windows_api", return_value=(kernel, psapi)):
            result = resources._process_memory(11)
        self.assertEqual(result["rss_bytes"], 101)
        self.assertEqual(result["private_commit_bytes"], 202)
        self.assertEqual(result["peak_working_set_bytes"], 303)
        self.assertEqual(result["peak_pagefile_usage_bytes"], 404)
        kernel.CloseHandle.assert_called_once_with(123)

    def test_process_memory_failure_still_closes_handle(self):
        kernel, psapi = Mock(), Mock()
        kernel.OpenProcess.return_value = 123
        psapi.GetProcessMemoryInfo.return_value = False
        with patch.object(resources, "_windows_api", return_value=(kernel, psapi)), patch.object(
            resources, "_win_error", return_value=OSError("模拟失败")
        ):
            with self.assertRaises(OSError):
                resources._process_memory(11)
        kernel.CloseHandle.assert_called_once_with(123)

    def test_snapshot_api_enumerates_and_closes_handle(self):
        kernel = Mock()
        kernel.CreateToolhelp32Snapshot.return_value = 456
        entries = iter([(10, 1), (20, 10)])

        def advance(handle, pointer):
            try:
                pid, parent = next(entries)
            except StopIteration:
                return False
            entry = ctypes.cast(pointer, ctypes.POINTER(resources._ProcessEntry32W)).contents
            entry.th32ProcessID = pid
            entry.th32ParentProcessID = parent
            return True

        kernel.Process32FirstW.side_effect = advance
        kernel.Process32NextW.side_effect = advance
        with patch.object(resources, "_windows_api", return_value=(kernel, Mock())), patch.object(
            resources.ctypes, "get_last_error", return_value=18, create=True
        ):
            self.assertEqual(resources._snapshot_parent_pids(), {10: 1, 20: 10})
        kernel.CloseHandle.assert_called_once_with(456)

    def test_probe_keeps_valid_system_data_but_marks_unknown_tree_failed(self):
        with patch.object(resources, "_system_memory", return_value={
            "ram_available_bytes": 10,
            "commit_available_bytes": 20,
        }), patch.object(resources, "_sample_process_tree", side_effect=OSError("后代已退出")):
            result = resources.probe_resources()
        self.assertFalse(result["ok"])
        self.assertEqual(result["ram_available"], 10)
        self.assertEqual(result["commit_available"], 20)
        self.assertIsNone(result["process_tree"])
        self.assertTrue(result["errors"])

    def test_missing_root_or_failed_child_is_not_silently_ignored(self):
        with patch.object(resources, "_snapshot_parent_pids", return_value={20: 10}):
            with self.assertRaises(OSError):
                resources._sample_process_tree(10)
        with patch.object(resources, "_snapshot_parent_pids", return_value={10: 1, 20: 10}), patch.object(
            resources, "_process_memory", side_effect=OSError("读取失败")
        ):
            with self.assertRaises(OSError):
                resources._sample_process_tree(10)


class MonitorTests(unittest.TestCase):
    def test_context_writes_initial_and_final_report_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resources.json"
            with patch.object(resources, "probe_resources", return_value=sample()), patch.object(
                resources.os, "replace", wraps=resources.os.replace
            ) as replace:
                with resources.MemoryMonitor(path) as monitor:
                    self.assertTrue(path.exists())
                    self.assertFalse(monitor.stop_requested)
                    self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["active"])
                    self.assertEqual(monitor.report["sample_interval_seconds"], 2)
                final = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(final["active"])
            self.assertEqual(final["sample_count"], 2)
            self.assertEqual(replace.call_count, 2)
            self.assertIsNotNone(final["finished_at_utc"])
            self.assertFalse(monitor._thread.is_alive())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_background_sampling_occurs_without_busy_wait(self):
        sampled_twice = threading.Event()
        counter = 0

        def probe():
            nonlocal counter
            counter += 1
            if counter >= 2:
                sampled_twice.set()
            return sample()

        with tempfile.TemporaryDirectory() as directory, patch.object(
            resources, "probe_resources", side_effect=probe
        ):
            with resources.MemoryMonitor(Path(directory) / "resources.json", interval=0.01) as monitor:
                self.assertTrue(sampled_twice.wait(timeout=2), "后台采样未按时发生。")
            self.assertGreaterEqual(monitor.report["sample_count"], 3)

    def test_exact_active_thresholds_allowed_but_one_byte_below_requests_stop(self):
        cases = [
            (3 * resources.GIB, 512 * resources.MIB, False),
            (3 * resources.GIB - 1, 512 * resources.MIB, True),
            (3 * resources.GIB, 512 * resources.MIB - 1, True),
        ]
        for ram, commit, should_stop in cases:
            with self.subTest(ram=ram, commit=commit), tempfile.TemporaryDirectory() as directory:
                with patch.object(resources, "probe_resources", return_value=sample(ram, commit)):
                    with resources.MemoryMonitor(Path(directory) / "resources.json") as monitor:
                        self.assertEqual(monitor.stop_requested, should_stop)
                self.assertEqual(bool(monitor.report["safety_stop_reasons"]), should_stop)

    def test_stop_is_sticky_and_report_is_a_copy(self):
        snapshots = [sample(ram=1), sample(), sample()]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            resources, "probe_resources", side_effect=snapshots
        ):
            with resources.MemoryMonitor(Path(directory) / "resources.json") as monitor:
                monitor._sample_once()
                altered = monitor.report
                altered["stop_requested"] = False
                altered["safety_stop_reasons"].clear()
                self.assertTrue(monitor.stop_requested)
                self.assertTrue(monitor.report["safety_stop_reasons"])
            self.assertTrue(monitor.stop_requested)

    def test_unknown_probe_or_malformed_sample_requests_stop(self):
        bad_samples = [{}, sample()]
        bad_samples[1]["process_tree"] = None
        for snapshot in bad_samples:
            with self.subTest(snapshot=snapshot), tempfile.TemporaryDirectory() as directory:
                with patch.object(resources, "probe_resources", return_value=snapshot):
                    with resources.MemoryMonitor(Path(directory) / "resources.json") as monitor:
                        self.assertTrue(monitor.stop_requested)
                        self.assertTrue(monitor.report["sample_errors"])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            resources, "probe_resources", side_effect=OSError("测试采样异常")
        ):
            with resources.MemoryMonitor(Path(directory) / "resources.json") as monitor:
                self.assertTrue(monitor.stop_requested)

    def test_records_minimum_available_snapshot_peaks_and_main_lifetime_peaks(self):
        first = sample(ram=7 * resources.GIB, commit=8 * resources.GIB, rss=110, private=120)
        second = sample(ram=8 * resources.GIB, commit=7 * resources.GIB, rss=90, private=140)
        second["process_tree"]["child_pids"] = [203, 204]
        second["process_tree"]["children_snapshot_count"] = 2
        second["process_tree"]["children_snapshot_rss_bytes"] = 50
        second["process_tree"]["main_process"]["peak_working_set_bytes"] = 700
        with tempfile.TemporaryDirectory() as directory, patch.object(
            resources, "probe_resources", side_effect=[first, second, second]
        ):
            with resources.MemoryMonitor(Path(directory) / "resources.json") as monitor:
                monitor._sample_once()
            report = monitor.report
        self.assertEqual(report["min_ram_available_bytes"], 7 * resources.GIB)
        self.assertEqual(report["min_commit_available_bytes"], 7 * resources.GIB)
        self.assertEqual(report["peak_tree_rss_bytes"], 110)
        self.assertEqual(report["peak_tree_private_commit_bytes"], 140)
        self.assertEqual(report["peak_children_snapshot_count"], 2)
        self.assertEqual(report["peak_children_snapshot_rss_bytes"], 50)
        self.assertEqual(report["children_seen_pid_count"], 3)
        self.assertEqual(report["main_lifetime_peak_working_set_bytes"], 700)
        self.assertEqual(report["main_lifetime_peak_pagefile_usage_bytes"], 600)
        self.assertTrue(report["measurement_limitations"])

    def test_write_failure_preserves_previous_json_and_requests_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resources.json"
            path.write_text('{"previous": true}', encoding="utf-8")
            with patch.object(resources, "probe_resources", return_value=sample()), patch.object(
                resources.os, "replace", side_effect=PermissionError("模拟占用")
            ):
                with resources.MemoryMonitor(path) as monitor:
                    self.assertTrue(monitor.stop_requested)
                    self.assertTrue(monitor.report["write_errors"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"previous": True})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_context_exception_propagates_and_final_report_is_written(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            resources, "probe_resources", return_value=sample()
        ):
            path = Path(directory) / "resources.json"
            with self.assertRaisesRegex(RuntimeError, "模拟训练异常"):
                with resources.MemoryMonitor(path):
                    raise RuntimeError("模拟训练异常")
            self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["active"])

    def test_interval_validation_and_reentry(self):
        for interval in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                resources.MemoryMonitor("resources.json", interval=interval)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            resources, "probe_resources", return_value=sample()
        ):
            monitor = resources.MemoryMonitor(Path(directory) / "resources.json")
            with monitor:
                with self.assertRaises(RuntimeError):
                    monitor.__enter__()


if __name__ == "__main__":
    unittest.main()
