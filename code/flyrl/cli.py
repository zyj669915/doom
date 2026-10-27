"""显式子命令；默认绝不启动训练。"""
import argparse
import json
from . import runner


def main():
    parser = argparse.ArgumentParser(description="冻结 MaleCNS + 小读出闭环实验（非生物学习复现）")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "smoke", "diagnose", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--arm", choices=("real", "rewired", "retinal", "pixels"))
        sub.add_argument("--algorithm", choices=("dqn", "ppo"))
        sub.add_argument("--seed", type=int)
        sub.add_argument("--rewire-seed", type=int)
        if name == "smoke":
            sub.add_argument("--steps", type=int, default=4)
        if name == "diagnose":
            sub.add_argument("--frames", type=int, default=8)
    sub = commands.add_parser("evaluate")
    sub.add_argument("--checkpoint", required=True, help="checkpoint-XXXXXXXX 目录，不接受旧模型")
    sub.add_argument("--split", choices=("val", "test"), default="val")
    sub.add_argument("--intervention", choices=("none", "black", "scramble"), default="none")
    sub.add_argument("--policy", choices=("learned", "random", "untrained"), default="learned")
    sub.add_argument("--episodes", type=int, default=None, help="默认每个唯一layout一局（combat每seed一局），禁止伪重复")
    sub = commands.add_parser("compare")
    sub.add_argument("results", nargs="+", help="冻结评估的 result.json")
    args = parser.parse_args()
    if args.command in ("doctor", "smoke", "diagnose", "train"):
        c = runner.load_config(args.config, {"arm": args.arm, "algorithm": args.algorithm,
                                            "seed": args.seed, "rewire_seed": args.rewire_seed})
        if args.command == "smoke":
            result = runner.smoke(c, args.steps)
        elif args.command == "diagnose":
            from .diagnostics import diagnose
            result = diagnose(c, args.frames)
        else:
            result = getattr(runner, args.command)(c)
    elif args.command == "evaluate":
        result = runner.evaluate(args.checkpoint, args.split, args.episodes,
                                 intervention=args.intervention, policy=args.policy)
    else:
        result = runner.compare(args.results)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
