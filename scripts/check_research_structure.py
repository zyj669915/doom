"""只读检查科研目录、文档链接、提示词契约和任务状态，不导入实验代码。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


REQUIRED = (
    'AGENTS.md',
    'research/README.md', 'research/CHARTER.md', 'research/STATUS.md',
    'research/TASKS.md', 'research/EVIDENCE.md',
    'research/tasks/T010-short-turn-adapter.md',
    'research/protocols/README.md',
    'research/protocols/short-turn-env-v1.md',
    'research/protocols/short-turn-learning-v1.md',
    'research/experiments/README.md',
    'research/experiments/LEGACY-20261026-real50k.md',
    'research/experiments/ENV-20261117-short-turn.md',
    'research/decisions/README.md',
    'research/decisions/ADR-001-research-workspace.md',
    'research/templates/README.md', 'research/templates/task.md',
    'research/templates/protocol.md', 'research/templates/result.md',
    'research/templates/decision.md', 'research/templates/handoff.md',
    'prompts/README.md', 'prompts/00-context.md',
    'prompts/01-research-lead.md', 'prompts/02-experiment-design.md',
    'prompts/03-implementation.md', 'prompts/04-review.md',
    'prompts/05-controlled-execution.md', 'prompts/06-result-analysis.md',
    'prompts/07-literature-check.md', 'prompts/08-handoff.md',
)
PROMPT_SECTIONS = ('输入', '授权与约束', '输出', '验收', '停止条件')
TASK_STATES = {'pending', 'in_progress', 'blocked', 'done', 'deferred'}
PROTOCOL_STATES = {'draft', 'locked', 'implemented', 'completed', 'stopped'}
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^)\n]+)\)')
TASK = re.compile(r'^\|\s*(T\d{3})\s*\|\s*([^|]+?)\s*\|', re.MULTILINE)


def prose_lines(text):
    """跳过围栏代码，不把命令示例里的路径当正文链接。"""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            yield number, line


def check_links(file, text, *, strict_evidence=False):
    errors, warnings = [], []
    checked = 0
    for number, line in prose_lines(text):
        for match in LINK.finditer(line):
            target = match.group(1).strip()
            if target.startswith('<') and target.endswith('>'):
                target = target[1:-1]
            # 本项目生成的链接不含空格标题；仅支持简单 inline Markdown。
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            relative = Path(unquote(parsed.path))
            label = f'{file}:{number}: {target}'
            if relative.is_absolute():
                errors.append(f'应使用相对本地链接：{label}')
                continue
            checked += 1
            resolved = (file.parent / relative).resolve()
            if not resolved.exists():
                evidence = any(part.startswith('runs') for part in relative.parts)
                message = f'链接目标缺失：{label}'
                if evidence and not strict_evidence:
                    warnings.append(message + '（原始产物可能未随文本归档）')
                else:
                    errors.append(message)
    return checked, errors, warnings


def check_tasks(text, *, allow_active=False):
    rows = TASK.findall(text)
    errors, seen = [], set()
    if not rows:
        errors.append('任务板没有可识别的 Txxx 任务行')
    for task_id, state in rows:
        if task_id in seen:
            errors.append(f'任务ID重复：{task_id}')
        seen.add(task_id)
        if state not in TASK_STATES:
            errors.append(f'任务状态无效：{task_id}={state}')
        elif state == 'in_progress' and not allow_active:
            errors.append(f'收尾时仍有进行中任务：{task_id}')
    return errors


def check_prompt(text):
    headings = {line[3:].strip() for _, line in prose_lines(text)
                if line.startswith('## ')}
    return [f'提示词缺少章节：{section}' for section in PROMPT_SECTIONS
            if section not in headings]


def inspect(root, *, allow_active=False, strict_evidence=False):
    root = Path(root).resolve()
    errors, warnings = [], []
    for relative in REQUIRED:
        file = root / relative
        if not file.is_file():
            errors.append(f'必要文件缺失：{relative}')
        elif file.stat().st_size == 0:
            errors.append(f'必要文件为空：{relative}')

    files = sorted(set((root / 'research').rglob('*.md'))
                   | set((root / 'prompts').rglob('*.md')))
    files += [p for p in (root / 'AGENTS.md', root.parent / 'AGENTS.md')
              if p.is_file()]
    links_checked = 0
    for file in files:
        try:
            text = file.read_text(encoding='utf-8-sig')
        except (OSError, UnicodeError) as exc:
            errors.append(f'读取失败：{file}: {exc}')
            continue
        if '�' in text:
            errors.append(f'出现Unicode替代字符：{file}')
        count, link_errors, link_warnings = check_links(
            file, text, strict_evidence=strict_evidence)
        links_checked += count
        errors.extend(link_errors)
        warnings.extend(link_warnings)
        if file == root / 'research/TASKS.md':
            errors.extend(check_tasks(text, allow_active=allow_active))
        if file.parent == root / 'prompts' and re.match(r'^\d{2}-', file.name):
            errors.extend(f'{file.name}: {e}' for e in check_prompt(text))
        if (file.parent == root / 'research/protocols'
                and file.name != 'README.md'):
            match = re.search(r'状态：([a-z_]+)', text)
            if not match or match.group(1) not in PROTOCOL_STATES:
                errors.append(f'协议状态缺失或无效：{file.name}')

    # 旧文档仅验证新入口，不把历史坏链混入本次新结构验收。
    for relative in ('README.md', 'PROJECT_INTRO.md', 'docs/PROGRESS.md'):
        file = root / relative
        try:
            text = file.read_text(encoding='utf-8-sig')
            if not re.search(r'\]\([^)]*research/README\.md\)', text):
                errors.append(f'缺少研究入口导航：{relative}')
        except (OSError, UnicodeError) as exc:
            errors.append(f'读取入口失败：{relative}: {exc}')

    return {
        'scope': 'local research structure only; no game, brain, training or network',
        'root': str(root),
        'passed': not errors,
        'markdown_files_checked': len(files),
        'relative_links_checked': links_checked,
        'allow_active_tasks': allow_active,
        'strict_evidence': strict_evidence,
        'errors': errors,
        'warnings': warnings,
        'limitations': [
            '只检查本地路径存在性，不校验锚点或外部网页',
            '不重验原始证据内容、脑图、checkpoint或科学结论',
            '仅扫描research/prompts/AGENTS；旧文档只检查新导航',
        ],
    }


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--allow-active', action='store_true',
                        help='开发期间允许in_progress，交接验收不要使用')
    parser.add_argument('--strict-evidence', action='store_true',
                        help='原始run链接缺失也作为错误；缺省只警告')
    args = parser.parse_args()
    report = inspect(args.root, allow_active=args.allow_active,
                     strict_evidence=args.strict_evidence)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
