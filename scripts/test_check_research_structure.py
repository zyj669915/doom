"""科研结构检查器的轻量回归，仅使用标准库与临时目录。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    'research_structure_checker',
    Path(__file__).with_name('check_research_structure.py'))
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ResearchStructureTests(unittest.TestCase):
    def test_task_states_and_duplicate_ids(self):
        self.assertEqual([], checker.check_tasks(
            '| T001 | done | 环境 |\n| T010 | pending | 适配 |'))
        self.assertTrue(checker.check_tasks('| T001 | unknown | x |'))
        self.assertTrue(checker.check_tasks(
            '| T001 | done | x |\n| T001 | pending | y |'))

    def test_active_tasks_require_explicit_development_flag(self):
        text = '| T000 | in_progress | 当前开发 |'
        self.assertTrue(checker.check_tasks(text))
        self.assertEqual([], checker.check_tasks(text, allow_active=True))

    def test_empty_task_board_is_rejected(self):
        self.assertTrue(checker.check_tasks('# 没有任务行'))

    def test_prompt_sections_and_fenced_examples(self):
        text = '\n'.join('## ' + s for s in checker.PROMPT_SECTIONS)
        self.assertEqual([], checker.check_prompt(text))
        self.assertTrue(checker.check_prompt('```text\n' + text + '\n```'))
        self.assertTrue(checker.check_prompt(text.replace('## 验收', '## 其他')))

    def test_existing_unicode_link_and_missing_document(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            file = root / 'source.md'
            (root / '研究目标.md').write_text('# 目标', encoding='utf-8')
            count, errors, warnings = checker.check_links(
                file, '[目标](%E7%A0%94%E7%A9%B6%E7%9B%AE%E6%A0%87.md#section)')
            self.assertEqual(1, count)
            self.assertEqual([], errors + warnings)
            _, errors, _ = checker.check_links(file, '[缺失](missing.md)')
            self.assertEqual(1, len(errors))

    def test_evidence_missing_is_warning_or_strict_error(self):
        with tempfile.TemporaryDirectory() as name:
            file = Path(name) / 'source.md'
            text = '[结果](runs_missing/result.json)'
            _, errors, warnings = checker.check_links(file, text)
            self.assertEqual([], errors)
            self.assertEqual(1, len(warnings))
            _, errors, warnings = checker.check_links(
                file, text, strict_evidence=True)
            self.assertEqual(1, len(errors))
            self.assertEqual([], warnings)

    def test_fences_external_and_fragment_links_are_not_local_checks(self):
        with tempfile.TemporaryDirectory() as name:
            file = Path(name) / 'source.md'
            text = ('```text\n[示例](missing.md)\n```\n'
                    '~~~\n[示例](also-missing.md)\n~~~\n'
                    '[外部](https://example.com/a) [章节](#title)')
            self.assertEqual((0, [], []), checker.check_links(file, text))

    def test_minimal_complete_fixture_and_missing_required_file(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / 'experiment'
            root.mkdir()
            for relative in checker.REQUIRED:
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                text = '# 文档\n'
                if relative == 'research/TASKS.md':
                    text += '| T001 | done | 完成 |\n'
                if relative.startswith('prompts/0'):
                    text += '\n'.join('## ' + s for s in checker.PROMPT_SECTIONS)
                if relative.startswith('research/protocols/short-'):
                    text += '状态：draft。\n'
                file.write_text(text, encoding='utf-8')
            for relative in ('README.md', 'PROJECT_INTRO.md', 'docs/PROGRESS.md'):
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('[科研](research/README.md)', encoding='utf-8')
            self.assertTrue(checker.inspect(root)['passed'])
            (root / 'research/CHARTER.md').unlink()
            self.assertFalse(checker.inspect(root)['passed'])

    def test_invalid_protocol_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            file = root / 'research/protocols/custom.md'
            file.parent.mkdir(parents=True)
            file.write_text('状态：unknown。', encoding='utf-8')
            errors = checker.inspect(root)['errors']
            self.assertTrue(any('协议状态' in error for error in errors))

    def test_entry_navigation_is_required(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'README.md').write_text('# 旧入口', encoding='utf-8')
            errors = checker.inspect(root)['errors']
            self.assertTrue(any('缺少研究入口导航：README.md' in e for e in errors))


if __name__ == '__main__':
    unittest.main(verbosity=2)
