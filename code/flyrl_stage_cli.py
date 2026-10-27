"""支持-I的独立入口；在导入numpy/torch/脑之前执行资源门禁。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyrl_stage.resources import gate, probe_resources

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='完整连接组三arm限额训练；资源不足拒绝启动')
    parser.add_argument('command', choices=('doctor', 'validate', 'task-probe', 'accept', 'train', 'evaluate', 'suite'))
    parser.add_argument('--config')
    parser.add_argument('--arm', choices=('real', 'rewired', 'frontend'), default='real')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--max-game-tics', type=int)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--policy', choices=('learned', 'random', 'untrained'), default='learned')
    parser.add_argument('--output', default='experiment/runs_stage_20261026')
    args = parser.parse_args()
    output = Path(args.output)
    output = output if output.is_absolute() else ROOT / output
    raw = {} if not args.config else json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if not isinstance(raw, dict):
        parser.error('配置必须为JSON对象')
    # 不允许通过入口把尚未测得峰值的任务预算调小来绕过系统余量。
    ram = max(1.5, raw.get('ram_gib', 1.5))
    commit = max(2., raw.get('commit_gib', 2.))
    admission = gate(ram, commit)
    check_path = output / ('preflight-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.json')
    write_json(check_path, {'command': args.command, 'admission': admission, 'report': str(check_path),
                            'started': False, 'stop': 'preflight_passed' if admission['allowed'] else 'resource_gate_rejected'})
    if args.command == 'doctor' or not admission['allowed']:
        print(json.dumps({'report': str(check_path), **admission}, ensure_ascii=False, indent=2))
        return 0 if admission['allowed'] else 3

    # 所有实际执行命令共享单槽；异常退出保留锁，拒绝悄悄并发。
    # suite父进程始终只用标准库，避免额外常驻一份torch。
    lock = output / 'ACTIVE.lock'
    with open(lock, 'x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'command': args.command}, stream)
    try:
        if args.command == 'suite':
            return suite(args, output)
        from flyrl_stage.config import load_config
        overrides = {'arm': args.arm, 'output': str(output), 'ram_gib': ram, 'commit_gib': commit}
        if args.max_game_tics is not None:
            overrides['max_game_tics'] = args.max_game_tics
        if args.command == 'accept':
            overrides.update(max_game_tics=256, wall_hours=1/6)
        c = load_config(args.config, overrides)
        if args.command == 'validate':
            print(json.dumps({'valid': True, 'protocol': c['protocol'], 'arm': c['arm']}, ensure_ascii=False))
            return 0
        from flyrl_stage.runner import train, evaluate, task_probe
        if args.command == 'task-probe':
            result = task_probe(c)
        elif args.command == 'evaluate':
            if not args.checkpoint:
                parser.error('evaluate需要--checkpoint')
            result = evaluate(args.checkpoint, quick=args.quick, policy=args.policy)
        else:
            result = train(c, resume=args.resume, smoke=args.command == 'accept')
        write_json(output / 'last-command.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('stop') in ('resource_gate_rejected', 'resource_stop', 'wall_limit', 'controlled_stop'):
            return 3
        if args.command == 'evaluate' and not result.get('complete'):
            return 3
        return 0
    finally:
        # 只清理由本次命令创建并持有的锁，不清理任何历史run。
        lock.unlink()


def suite(args, output):
    """一个进程只持有一个脑；子命令退出释放后才执行下一项。"""
    # suite持有全局锁；子任务在独立子目录中取得自己的锁。
    script = str(Path(__file__).resolve())
    state_path = output / 'suite-status.json'
    state = {'protocol': 'malecns-near-turn-three-arm-v1', 'completed': [], 'active': None}

    def invoke(command, arm, *extra):
        if (output / 'STOP').exists():
            raise RuntimeError('suite收到STOP；未启动下一项')
        arm_output = output / arm
        cmd = [sys.executable, '-I', '-B', script, command, '--arm', arm,
               '--output', str(arm_output), *extra]
        if args.config:
            cmd.extend(['--config', args.config])
        state['active'] = {'command': command, 'arm': arm, 'extra': list(extra)}
        write_json(state_path, state)
        log = output / f'{arm}-{command}-{uuid.uuid4().hex[:8]}.log'
        with open(log, 'w', encoding='utf-8') as stream:
            proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'子任务退出码{proc.returncode}；保留日志：{log}')
        result = json.loads((arm_output / 'last-command.json').read_text(encoding='utf-8'))
        state['completed'].append({'command': command, 'arm': arm, 'run': result['run'], 'log': str(log)})
        write_json(state_path, state)
        return result

    try:
        probe = invoke('task-probe', 'frontend')
        if not probe.get('acceptable_for_pilot'):
            raise RuntimeError('任务负对照未通过；不投入完整脑训练')
        for arm in ('real', 'rewired', 'frontend'):
            accepted = invoke('accept', arm)
            if accepted['game_tics'] + accepted['initial_observation_tics'] != 256:
                raise RuntimeError('资源验收未达到256tic，不开始正式训练')
        for arm in ('real', 'rewired', 'frontend'):
            resume = None
            evaluated = set()
            for budget in (10000, 30000, 50000):
                extra = ['--max-game-tics', str(budget)]
                if resume:
                    extra.extend(['--resume', resume])
                result = invoke('train', arm, *extra)
                if result['game_tics'] != budget or not result['optimized']:
                    raise RuntimeError('未完成预锁样本预算或无优化，不继续下一阶段')
                resume = result['checkpoint']
                queue = Path(result['run']) / 'evaluation-queue.jsonl'
                for line in queue.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    point = row['game_tics']
                    if point in evaluated:
                        continue
                    invoke('evaluate', arm, '--checkpoint', row['checkpoint'],
                           *(['--quick'] if row['quick'] else []))
                    if point == 0:
                        invoke('evaluate', arm, '--checkpoint', row['checkpoint'], '--quick', '--policy', 'random')
                    evaluated.add(point)
        state.update(active=None, stop='first_screen_complete')
        write_json(state_path, state)
        return 0
    except BaseException as exc:
        state.update(active=None, stop='blocked_or_failed', error=str(exc))
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    sys.exit(main())
