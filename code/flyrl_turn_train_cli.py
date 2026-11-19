"""独立短任务训练CLI；标准库准入先于torch/NumPy/游戏导入，无自动队列。"""
import argparse
import json
import os
import signal
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyrl_turn_train.config import load_config, resolve, ROOT, PROTOCOL


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'doctor', 'train', 'accept', 'evaluate'))
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--policy', default='learned', choices=(
        'learned', 'untrained', 'random', 'sequence-negative', 'sequence-positive'))
    parser.add_argument('--intervention', choices=('none', 'black'), default='none')
    args = parser.parse_args(argv)
    if args.resume and args.command != 'train':
        parser.error('--resume仅用于train')
    if args.command == 'evaluate' and not args.checkpoint:
        parser.error('evaluate需要--checkpoint')
    if args.checkpoint and args.command != 'evaluate':
        parser.error('--checkpoint仅用于evaluate')
    if (args.quick or args.policy != 'learned' or args.intervention != 'none') and args.command != 'evaluate':
        parser.error('评估参数不能应用于训练')
    # 原配置仅读取一次；门禁与执行均使用已校验的同一快照。
    if args.config:
        raw = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    elif args.command == 'evaluate':
        from flyrl_turn_train.storage import read_manifest
        raw = read_manifest(args.checkpoint)['config']
    else:
        raw = {}
    c = load_config(values=raw)
    if args.output:
        c['output'] = args.output
    c['output'] = str(resolve(c['output']))
    c = load_config(values=c)
    if args.command == 'validate':
        print(json.dumps({'valid': True, 'protocol': PROTOCOL, 'arm': c['arm'],
                          'commands': ['train', 'accept', 'evaluate'],
                          'conditions': {'train': 12, 'dev': 12, 'confirm': 24},
                          'evaluation_split_supported': ['dev'],
                          'max_game_tics': c['max_game_tics'],
                          'research_protocol_locked': False,
                          'training_started': False}, ensure_ascii=False, indent=2))
        return 0
    from flyrl_stage.resources import gate, MemoryMonitor
    started = time.monotonic()
    admission = gate(c['ram_gib'], c['commit_gib'])
    output = Path(c['output'])
    path = output / (time.strftime('%Y%m%d-%H%M%S') + f'-{args.command}-{c["arm"]}-' + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'config.json', c)
    write_json(path / 'admission.json', admission)
    if args.command == 'doctor' or not admission['allowed']:
        result = {'run': str(path), 'allowed': admission['allowed'], 'errors': admission['errors'],
                  'stop': 'preflight_passed' if admission['allowed'] else 'resource_gate_rejected',
                  'started': False, 'training_started': False, 'complete': False}
        write_json(path / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if admission['allowed'] else 3
    # 与output无关的固定锁，避免换输出目录意外并发两个新适配器。
    lock = ROOT / 'experiment/TURN_TRAIN_ACTIVE.lock'
    try:
        stream = open(lock, 'x', encoding='utf-8')
    except FileExistsError:
        result = {'run': str(path), 'stop': 'active_slot_locked', 'started': False,
                  'training_started': False, 'complete': False, 'lock': str(lock)}
        write_json(path / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 3
    with stream:
        json.dump({'pid': os.getpid(), 'run': str(path)}, stream)
    seconds = (c['eval_wall_hours'] * 3600 if args.command == 'evaluate' else
               min(600., c['wall_hours'] * 3600) if args.command == 'accept' else c['wall_hours'] * 3600)
    interrupted = [False]
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    try:
        with MemoryMonitor(path / 'resources.json') as monitor:
            monitor.interrupt_requested = lambda: interrupted[0]
            # 检查STOP/墙钟也在导入大库前，导入时间计入本次预算。
            early = ('controlled_stop' if interrupted[0] else
                     'resource_stop' if monitor.stop_requested else
                     'controlled_stop' if any((p / 'STOP').exists() for p in (path, output, output.parent)) else
                     'wall_limit' if time.monotonic() - started >= seconds else None)
            if early:
                result = {'run': str(path), 'stop': early, 'complete': False,
                          'started': False, 'training_started': False, 'passed': False}
            else:
                from flyrl_turn_train.runner import run_train, run_evaluate
                if args.command == 'evaluate':
                    result = run_evaluate(c, path, monitor, args.checkpoint, started=started,
                                          quick=args.quick, policy=args.policy, intervention=args.intervention)
                else:
                    result = run_train(c, path, monitor, started=started, resume=args.resume,
                                       smoke=args.command == 'accept')
            write_json(path / 'result.json', result)
        # 最后close/快照/持久化及monitor.__exit__的停止仍否决complete。
        reason = ('controlled_stop' if interrupted[0] else
                  'resource_stop' if monitor.stop_requested else
                  'controlled_stop' if any((p / 'STOP').exists() for p in (path, output, output.parent)) else
                  'wall_limit' if result.get('previous_wall_s', 0.) + time.monotonic() - started >= seconds else None)
        if reason:
            result.update(stop=reason, complete=False, passed=False)
            if args.command == 'evaluate':
                result['success_rate'] = None
            write_json(path / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get('complete') else 3
    except BaseException as exc:
        write_json(path / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                  'latest_checkpoint_pointer': str(path / 'latest-checkpoint.json')})
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
