# -*- coding: utf-8 -*-
"""闭环早期行为诊断：50 帧。记录每帧 turn 预测、位置、特征漂移。
"""
import math
import sys
import numpy as np
import torch
import vizdoom as vzd
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain  # noqa: E402
from doom.game import Game, retinal_samples  # noqa: E402
from readout_experiment import agc, motion, custom_step, SCALE, BIAS, THRESH, DUR_MS, BRAIN  # noqa: E402
from corridor_teacher import head_to_action  # noqa: E402
from corridor_student_win import MLPWin, CK, UV  # noqa: E402

ck = torch.load(CK, map_location='cpu', weights_only=False)
idx, mu, sd = ck['topk'], ck['mu'], ck['sd']
model = MLPWin(len(idx) * ck['win'] * 2)
model.load_state_dict(ck['state'])
model.eval()
d = len(idx)
WIN = ck['win']

brain = Brain(BRAIN)
brain.weight = (brain.weight * SCALE).astype(np.float32)
game = Game(seed=41027, scenario='corridor_avoid')
game.new_episode()
prev = None
buf = np.zeros((WIN, d), np.float32)
buf_prev = np.zeros((WIN, d), np.float32)
ptr = 0
x0, y0 = 128.0, 128.0
ti_seq = []
for t in range(50):
    rgb = game.pixels()
    lum = retinal_samples(rgb, UV).astype(np.float32)
    l2 = agc(motion(lum, prev))
    prev = lum
    c = custom_step(brain, l2, DUR_MS, bias=BIAS, thresh=THRESH)
    v = c[idx].astype(np.float32)
    buf[ptr % WIN] = v
    if t == 0:
        buf_prev[ptr % WIN] = np.zeros(d, np.float32)
    else:
        buf_prev[ptr % WIN] = buf[(ptr - 1) % WIN]
    ptr += 1
    feat = np.zeros(WIN * d * 2, np.float32)
    col = 0
    for j in range(WIN):
        i0 = (ptr - WIN + j) % WIN
        feat[col:col + d] = buf[i0]
        col += d
        feat[col:col + d] = buf[i0] - buf_prev[i0]
        col += d
    F = (feat - mu) / sd
    nrm = float(np.linalg.norm(F))
    with torch.no_grad():
        logits = model(torch.from_numpy(F).float().unsqueeze(0)).squeeze(0)
        probs = torch.softmax(logits, 0).numpy()
        ti = int(logits.argmax().item())
    ti_seq.append(ti)
    game.act(head_to_action(ti, 1))
    x1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_X))
    y1 = float(game.game.get_game_variable(vzd.GameVariable.POSITION_Y))
    moved = math.hypot(x1 - x0, y1 - y0)
    if t < 12 or t % 10 == 0:
        print(f't={t:3d} ti={ti} p={probs.round(2)} moved={moved:5.2f} x={x1:6.1f} y={y1:6.1f} '
              f'feat|F|={nrm:.1f}', flush=True)
    x0, y0 = x1, y1
import collections
print('turn seq counts:', collections.Counter(ti_seq), flush=True)
game.close()
