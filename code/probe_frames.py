"""Probe: look at actual combat_survival frames to design a working rule teacher.
Saves 4 frame snapshots + motion stats; prints color stats per frame.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from doom.game import Game

def gray(rgb): return rgb.mean(2).astype(np.float32)

g = Game(seed=41029, scenario='combat_survival')
prev = gray(g.pixels())
from PIL import Image
shots = []
for t in range(210):
    if g.game.is_episode_finished(): break
    rgb = g.pixels()
    cur = gray(rgb)
    d = np.abs(cur - prev)
    mask = d > 12.0
    n = int(mask.sum())
    print(f't={t:3d} motion_px={n:5d}', flush=True)
    if t in (10, 60, 110, 160):
        shots.append((t, rgb, d > 12.0))
        Image.fromarray(rgb).save(ROOT/'outputs'/'doom'/('probe_frame_%03d.png' % t))
    # stand still (no act) so enemies approach
    g.act({'turn': 0.0, 'forward': 0.0, 'attack': False})
    prev = cur
g.close()
print('saved probe frames:', [s[0] for s in shots])
