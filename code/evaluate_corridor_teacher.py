# -*- coding: utf-8 -*-
"""走廊老师 argmax 确定性终验（Windows 版）。

用法: python evaluate_corridor_teacher.py --seeds 5 --eps-per-seed 20 --max-tics 2000
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import vizdoom as vzd

sys.path.insert(0, r'D:\fly\doomfly')
from doom.game import Game  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corridor_teacher import (  # noqa: E402
    Policy, ObsBuilder, head_to_action, dist, at_exit, MAX_TICS, load_teacher,
)

CKPT = r'D:\fly\experiment\code\outputs\doom\corridor-teacher-v6\teacher-corridor-final.pt'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--eps-per-seed', type=int, default=20)
    p.add_argument('--max-tics', type=int, default=2000)
    args = p.parse_args()

    policy, meta = load_teacher(CKPT)
    use_angle = bool(meta.get('use_angle', True))

    results = {'seeds': args.seeds, 'eps_per_seed': args.eps_per_seed,
               'teacher_use_angle': use_angle,
               'exit_count': 0, 'episodes': 0, 'exit_tics': [], 'wall_rate': [],
               'max_prog': [], 'rew': [], 'per_seed_exit_rates': []}
    for s in range(args.seeds):
        game = Game(seed=41027 + s, scenario='corridor_avoid')
        ob = ObsBuilder(use_angle=use_angle)
        seed_exit = 0
        for _ in range(args.eps_per_seed):
            obs = ob.reset(game)
            done = False
            tics = 0
            rew = 0.0
            hits = 0
            max_prog = 0.0
            while not done and tics < args.max_tics:
                with torch.no_grad():
                    at, af, _, _ = policy.act(torch.from_numpy(obs).float().unsqueeze(0), greedy=True)
                    a = (at.item(), af.item())
                act = head_to_action(*a)
                x0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
                y0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
                r = game.act(act)
                x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
                y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
                moved = math.hypot(x1 - x0, y1 - y0)
                if act['forward'] > 0 and moved < 0.5:
                    r += -0.15
                    hits += 1
                if at_exit(x1, y1):
                    r += 10.0
                    done = True
                d0, d1 = dist(x0, y0), dist(x1, y1)
                rew += r + 0.01 * (d0 - d1)
                prog = 1.0 - dist(x1, y1) / 3456.0
                max_prog = max(max_prog, prog)
                tics += 1
                if game.game.is_episode_finished():
                    done = True
                else:
                    obs = ob.step(game)
            exited = 1 if (done and at_exit(x1, y1)) else 0
            seed_exit += exited
            results['exit_count'] += exited
            results['episodes'] += 1
            results['exit_tics'].append(tics if exited else args.max_tics)
            results['wall_rate'].append(hits / max(tics, 1))
            results['max_prog'].append(max_prog)
            results['rew'].append(rew)
            game.new_episode()
        game.close()
        results['per_seed_exit_rates'].append(seed_exit / args.eps_per_seed)

    n = results['episodes']
    exit_tics = [t for t, e in zip(results['exit_tics'], [1] * n)]
    print({
        'seeds': args.seeds,
        'eps_per_seed': args.eps_per_seed,
        'exit_rate': round(results['exit_count'] / n, 4),
        'exit_count': results['exit_count'],
        'episodes': n,
        'exit_tics_mean': round(float(np.mean(results['exit_tics'])), 1),
        'exit_tics_std': round(float(np.std(results['exit_tics'])), 1),
        'wall_rate': round(float(np.mean(results['wall_rate'])), 4),
        'max_prog_mean': round(float(np.mean(results['max_prog'])), 4),
        'rew_mean': round(float(np.mean(results['rew'])), 3),
        'per_seed_exit_rates': results['per_seed_exit_rates'],
    })


if __name__ == '__main__':
    main()
