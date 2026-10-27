# -*- coding: utf-8 -*-
"""决定性 A/B：老师策略驱动闭环 50 帧，检查特征范数是否回到训练水平（~71）。
若 |F|≈71 → 管线一致，学生闭环失败是分布偏移（BC 通病）；
若 |F|≈260 → 闭环管线与训练存在系统性差异（bug）。
"""
import sys
import numpy as np
import torch
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain  # noqa: E402
from doom.game import Game, retinal_samples  # noqa: E402
from readout_experiment import agc, motion, custom_step, SCALE, BIAS, THRESH, DUR_MS, BRAIN  # noqa: E402
from corridor_teacher import Policy, ObsBuilder, head_to_action  # noqa: E402
from corridor_readout_upgrade import select_topk, build  # noqa: E402
from corridor_student import zscore_local, OUT  # noqa: E402

UV = np.load(BRAIN)['uv']

TEACHER = CODE / 'outputs' / 'doom' / 'corridor-teacher-v6' / 'teacher-corridor-final.pt'
acts = np.load(OUT / 'acts.npy')
acts3 = (acts // 2).astype(np.int64)
counts = np.load(OUT / 'counts-real.npy')
idx = select_topk(counts, acts3, 512)

ck = torch.load(TEACHER, map_location='cpu')
policy = Policy(obs_dim=24 * 32 * 2 + 2)
policy.load_state_dict(ck)
policy.eval()

brain = Brain(BRAIN)
brain.weight = (brain.weight * SCALE).astype(np.float32)
game = Game(seed=41027, scenario='corridor_avoid')
game.new_episode()
ob = ObsBuilder()
prev = None
lum_seq = []
for t in range(50):
    rgb = game.pixels()
    lum = retinal_samples(rgb, UV).astype(np.float32)
    lum_seq.append(lum)
    o = ob.step(game) if t > 0 else ob.reset(game)
    with torch.no_grad():
        at, af, _, _ = policy.act(torch.from_numpy(o).float().unsqueeze(0), greedy=True)
    game.act(head_to_action(at.item(), af.item()))
lum_seq = np.array(lum_seq)
# 用与训练完全相同的 sim 流程重算 counts
brain2 = Brain(BRAIN)
brain2.weight = (brain2.weight * SCALE).astype(np.float32)
T = len(lum_seq)
counts2 = np.zeros((T, brain2.n), dtype=np.float32)
prev2 = None
for i, l in enumerate(lum_seq):
    l2 = agc(motion(l, prev2))
    prev2 = l
    c = custom_step(brain2, l2, DUR_MS, bias=BIAS, thresh=THRESH)
    counts2[i] = c.astype(np.float32)
F = zscore_local(build(counts2, idx, 5, True).astype(np.float32))
n = np.linalg.norm(F, axis=1)
print('teacher-driven closed-loop |F|: mean=%.1f std=%.1f p5=%.1f p95=%.1f' %
      (n.mean(), n.std(), np.percentile(n, 5), np.percentile(n, 95)))
print('train-set reference: 71.4 +- 4.8 (p95 76.8)')
game.close()
