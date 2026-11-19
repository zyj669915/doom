"""独立短程任务CLI；真实游戏命令在导入大库前门禁，不提供正式训练。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='短程左右转角任务：配置检查、资源检查、真实环境验收；不训练')
    parser.add_argument('command', choices=('validate', 'doctor', 'accept'))
    parser.add_argument('--config')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    raw = {} if args.config is None else json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if not isinstance(raw, dict):
        parser.error('配置必须是JSON对象')
    output = Path(args.output or raw.get('output', 'experiment/runs_turn_20261117'))
    output = output if output.is_absolute() else root / output
    if args.command == 'validate':
        from flyrl_turn.config import load_config, manifest
        c = load_config(values=raw)
        m = manifest()
        print(json.dumps({'valid': True, 'protocol': c['protocol'], 'conditions': {
            k: len(v) for k, v in m['splits'].items()}, 'training_supported': False}, ensure_ascii=False, indent=2))
        return 0
    ram, commit = raw.get('ram_gib', 1.5), raw.get('commit_gib', 2.)
    if (type(ram) not in (int, float) or not 1.5 <= ram < float('inf') or
            type(commit) not in (int, float) or not max(2., ram) <= commit < float('inf')):
        parser.error('任务RAM/commit预算不得降低或非有限')
    from flyrl_stage.resources import gate, MemoryMonitor
    admission = gate(ram, commit)
    path = output / (time.strftime('%Y%m%d-%H%M%S') + '-' + args.command + '-' + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'admission.json', admission)
    if args.command == 'doctor' or not admission['allowed']:
        result = {'run': str(path), 'allowed': admission['allowed'], 'errors': admission['errors'],
                  'stop': 'preflight_passed' if admission['allowed'] else 'resource_gate_rejected',
                  'started': False, 'training_started': False}
        write_json(path / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if admission['allowed'] else 3
    lock = output / 'ACTIVE.lock'
    with open(lock, 'x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'run': str(path)}, stream)
    started = time.monotonic()
    max_seconds = 1200
    try:
        with MemoryMonitor(path / 'resources.json') as monitor:
            if monitor.stop_requested:
                result = {'run': str(path), 'stop': 'resource_stop_before_import', 'passed': False}
            else:
                from flyrl_turn.config import load_config
                from flyrl_turn.acceptance import accept, check_stop
                c = load_config(values=raw)
                c['output'] = str(output)
                write_json(path / 'config.json', c)
                if monitor.stop_requested:
                    result = {'run': str(path), 'stop': 'resource_stop_before_game', 'passed': False}
                else:
                    result = accept(c, path, monitor, max_seconds=max_seconds, started=started)
            write_json(path / 'result.json', result)
        # 最终采样/持久化期间的资源、STOP和墙钟停止同样否决通过。
        if result.get('passed'):
            reason = check_stop(path, monitor, started, max_seconds)
            if reason:
                result.update(passed=False, stop=reason)
                write_json(path / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get('passed') else 3
    except BaseException as exc:
        write_json(path / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'training_started': False})
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
