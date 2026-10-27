"""显式命令；无参数不会开始训练。"""
import argparse
import json
from pathlib import Path
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description='多路线闭环训练；正式训练必须显式 train/suite run')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('doctor', 'train', 'smoke', 'diagnose'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        if name in ('train', 'smoke'):
            p.add_argument('--resume')
            p.add_argument('--steps', type=int)
            p.add_argument('--wall-seconds', type=float)
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--intervention', choices=['none', 'black', 'scramble', 'brain_zero', 'brain_permute'], default='none')
    p.add_argument('--policy', choices=['learned', 'random', 'untrained'], default='learned')
    p = sub.add_parser('compare')
    p.add_argument('results', nargs='+')
    p.add_argument('--mode', choices=['performance', 'intervention', 'connectome'], default='performance')
    p = sub.add_parser('stop')
    p.add_argument('--run', required=True)
    p = sub.add_parser('resume')
    p.add_argument('--run', required=True)
    p.add_argument('--config', required=True)
    p = sub.add_parser('suite')
    p.add_argument('action', choices=['plan', 'run', 'status'])
    p.add_argument('--config', required=True)
    p.add_argument('--parallel', action='store_true')
    args = parser.parse_args()
    if args.command in ('doctor', 'train', 'smoke', 'diagnose'):
        overrides = {'max_game_tics': 32, 'wall_hours': 180 / 3600} if args.command == 'smoke' else {}
        if getattr(args, 'steps', None) is not None:
            overrides['max_game_tics'] = args.steps
        if getattr(args, 'wall_seconds', None) is not None:
            overrides['wall_hours'] = args.wall_seconds / 3600
        c = load_config(args.config, overrides)
        from . import runner
        if args.command == 'doctor':
            result = {'config': c, 'provenance': runner.provenance(c)}
        elif args.command == 'diagnose':
            from .diagnostics import diagnose
            result = diagnose(c)
        else:
            result = runner.train(c, args.resume, smoke=args.command == 'smoke')
    elif args.command == 'evaluate':
        from .runner import evaluate
        result = evaluate(args.checkpoint, args.split, args.quick, args.intervention, args.policy)
    elif args.command == 'compare':
        from .evaluation import compare
        result = compare(args.results, args.mode)
    elif args.command == 'stop':
        from .suite import stop
        result = stop(args.run)
    elif args.command == 'resume':
        from .runner import train
        latest = json.loads((Path(args.run) / 'latest-checkpoint.json').read_text(encoding='utf-8'))
        result = train(load_config(args.config), latest['checkpoint'])
    else:
        from . import suite
        result = getattr(suite, args.action)(args.config, parallel=args.parallel) if args.action in ('run', 'plan') else suite.status(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
