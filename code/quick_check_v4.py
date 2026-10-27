# -*- coding: utf-8 -*-
"""v4 argmax 快速验证：1 seed × 5 eps，打印每 ep 结果与动作计数。"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import vizdoom as vzd

sys.path.insert(0, r'D:\fly\doomfly')
from doom.game import Game  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corridor_teacher import Policy, ObsBuilder, head_to_action, dist, at_exit  # noqa: E402

CKPT = r'D:\fly\experiment\code\outputs\doom\corridor-teacher-v6\teacher-corridor-final.pt'
policy = Policy(obs_dim=24 * 32 * 2 + 2)
policy.load_state_dict(torch.load(CKPT, map_location='cpu'))
policy.eval()

game = Game(seed=41027, scenario='corridor_avoid')
ob = ObsBuilder()
from collections import Counter
act_counter = Counter()
for ep in range(5):
    obs = ob.reset(game)
    tics, rew, hits = 0, 0.0, 0
    exited = False
    turn_seq = []
    while tics < 1500:
        with torch.no_grad():
            at, af, _, _ = policy.act(torch.from_numpy(obs).float().unsqueeze(0), greedy=True)
        act = head_to_action(at.item(), af.item())
        act_counter[(int(at), int(af))] += 1
        turn_seq.append(int(at))
        x0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
        y0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
        game.act(act)
        x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
        y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
        moved = math.hypot(x1 - x0, y1 - y0)
        if act['forward'] > 0 and moved < 0.5:
            rew -= 0.15
            hits += 1
        if at_exit(x1, y1):
            rew += 10.0
            exited = True
        rew += 0.01 * (dist(x0, y0) - dist(x1, y1))
        tics += 1
        if game.game.is_episode_finished():
            break
        obs = ob.step(game)
    # 转角区域动作统计 (x>=1500)
    print(f"ep={ep} exit={exited} tics={tics} rew={rew:.1f} hits={hits} "
          f"maxprog={1.0 - dist(x1,y1)/3456.0:.3f} pos=({x1:.0f},{y1:.0f})")
    game.new_episode()
game.close()
print("action counts:", dict(act_counter))
print("turn hist:", dict(Counter(turn_seq)))
