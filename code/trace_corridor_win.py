# -*- coding: utf-8 -*-
"""Windows 走廊 argmax 单 episode 轨迹诊断。"""
import sys
import math
from pathlib import Path

import numpy as np
import torch
import vizdoom as vzd

sys.path.insert(0, r'D:\fly\doomfly')
from doom.game import Game  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corridor_teacher import Policy, ObsBuilder, index_to_action, dist, at_exit  # noqa: E402

CKPT = r'D:\fly\experiment\code\outputs\doom\corridor-teacher-v3\teacher-corridor-final.pt'
policy = Policy(obs_dim=24 * 32 * 2 + 2)
policy.load_state_dict(torch.load(CKPT, map_location='cpu'))
policy.eval()

game = Game(seed=41027, scenario='corridor_avoid')
ob = ObsBuilder()
obs = ob.reset(game)
for t in range(120):
    if game.game.is_episode_finished():
        print(f't={t} EPISODE FINISHED')
        break
    with torch.no_grad():
        a = int(policy.act(torch.from_numpy(obs).float().unsqueeze(0), greedy=True)[0].item())
    act = index_to_action(a)
    x0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
    y0 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
    ang0 = float(game.game.get_game_variable(vzd.GameVariable.ANGLE))
    game.act(act)
    x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
    y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
    ang1 = float(game.game.get_game_variable(vzd.GameVariable.ANGLE))
    moved = math.hypot(x1 - x0, y1 - y0)
    flag = ''
    if act['forward'] > 0 and moved < 0.5:
        flag = 'WALL'
    if t < 40 or t % 10 == 0 or flag:
        print(f"t={t:3d} act=t{act['turn']:+d}/f{act['forward']:2d} pos=({x1:7.1f},{y1:7.1f}) "
              f"ang={ang1:6.1f} moved={moved:5.2f} d={dist(x1,y1):6.0f} {flag}")
    obs = ob.step(game)
    if at_exit(x1, y1):
        print(f't={t} EXIT! pos=({x1:.1f},{y1:.1f})')
        break
game.close()
