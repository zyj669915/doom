# -*- coding: utf-8 -*-
"""Corrected fly vision: the fly should see AROUND, not just the player's 90 deg
screen.  We collect a 270-deg panorama (left | front | right, each ~90 deg) and
re-project the 3,335 photoreceptors onto it as a compound-eye view:
  L eye -> panorama left half (left 135 deg), R eye -> right half (right 135 deg),
  with a 10% central binocular overlap, no planar 90-deg restriction.

Panels:
  1. the 270-deg panorama as a human sees it
  2. compound-eye mosaic at ~real ommatidia resolution (each ommatidium cell
     shows the mean luminance of the photoreceptors mapped into it)
  3. photoreceptor point cloud + injected current (what drives the brain)

This is a display/projection fix of the sensory mapping; graph.npz uv is left
untouched (a runtime uv variant).  Saving it back as the training uv is a
separate, explicit step.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyh.ttf', r'C:\Windows\Fonts\simhei.ttf']:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

import vizdoom
from doom.game import Game, retinal_samples

ROOT = Path(r'D:\fly\doomfly')
GRAPH = ROOT / 'outputs' / 'doom' / 'malecns_v1' / 'graph.npz'
OUT_DIR = Path(r'D:\fly\reference\fly-vision')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAN_W = 3 * 640          # 1920
PAN_H = 480
TURN_TARGET_DEG = 90.0

def collect_panorama(seed=41027):
    """Collect left|front|right views around the same spot (approx. simultaneous:
    turns advance the game a few ticks; acceptable for a visualisation)."""
    game = Game(seed=seed, scenario='combat_survival')
    game.new_episode()

    def grab():
        return game.pixels()

    def turn_toward(delta_deg):
        # read current angle, hold turn button until we have moved |delta_deg|
        start = float(game.game.get_game_variable(vizdoom.GameVariable.ANGLE))
        target = (start + delta_deg) % 360.0
        moved = 0.0
        while moved < abs(delta_deg):
            act = {'turn': np.sign(delta_deg) * 6, 'forward': 0, 'attack': 0}
            game.act(act)
            now = float(game.game.get_game_variable(vizdoom.GameVariable.ANGLE))
            moved = abs((now - start + 540) % 360 - 180)

    front0 = grab()                                   # facing initial direction
    turn_toward(-TURN_TARGET_DEG); left = grab()      # look left 90
    turn_toward(+TURN_TARGET_DEG);                    # back to front
    turn_toward(+TURN_TARGET_DEG); right = grab()     # look right 90
    turn_toward(-TURN_TARGET_DEG)                     # back to front
    game.close()
    # panorama order: LEFT | FRONT | RIGHT  (1920x480)
    return np.concatenate([left, front0, right], axis=1)

def corrected_uv(uv):
    """Re-map the planar uv onto the 270-deg panorama:
    L eye -> panorama x in [0,0.55], R eye -> [0.45,1.0] (10% binocular overlap),
    y unchanged."""
    uv2 = uv.copy().astype(np.float32)
    L = uv[:, 0] < 0.5
    uv2[L, 0] = uv[L, 0] * (0.55 / 0.6)                 # L: 0..0.6 -> 0..0.55
    uv2[~L, 0] = 0.45 + (uv[~L, 0] - 0.4) * (0.55 / 0.6)  # R: 0.4..1.0 -> 0.45..1.0
    return uv2

def agc(lum, eps=1e-4):
    sd = lum.std()
    if sd < eps:
        return lum.copy()
    return 0.5 + 0.5 * np.tanh((lum - lum.mean()) / (sd + eps) / 1.5)

def main():
    g = np.load(GRAPH)
    uv = g['uv'].astype(np.float32)
    print('collecting 270-deg panorama ...', flush=True)
    pan = collect_panorama()
    print('panorama shape:', pan.shape, flush=True)

    uv_c = corrected_uv(uv)
    lum = retinal_samples(pan, uv_c).astype(np.float32)   # 3335 photoreceptor reads
    drive = 30 * lum / (0.02 + lum)
    lum_agc = agc(lum)                                      # contrast-stretched (light adapt)

    # ---- panel 2: compound-eye mosaic at ommatidia-like resolution ----
    nx, ny = 96, 30           # ~2880 cells; keep cell size ~20px
    cells = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
    ux = np.clip(uv_c[:, 0], 0, 1 - 1e-6)
    uy = np.clip(uv_c[:, 1], 0, 1 - 1e-6)
    cx = (ux * nx).astype(int); cy = (uy * ny).astype(int)
    for i in range(len(lum)):
        cells[cy[i], cx[i]] += lum[i]; cnt[cy[i], cx[i]] += 1
    cells = np.divide(cells, cnt, out=np.zeros_like(cells), where=cnt > 0)
    mosaic = np.zeros((ny, nx, 3), np.float32)
    mosaic[:, :, 0] = mosaic[:, :, 1] = mosaic[:, :, 2] = cells

    fig, axes = plt.subplots(1, 4, figsize=(28, 5.2), dpi=110)
    fig.suptitle('修正后：果蝇看周围 (270° 全景, combat_survival seed 41027)',
                 fontsize=13, color='#111')

    ax = axes[0]
    ax.imshow(pan)
    ax.set_title('① 270° 全景：左 | 前 | 右（人眼视角）', fontsize=11)
    ax.axis('off')

    ax = axes[1]
    ax.imshow(mosaic)
    ax.set_title('② 复眼小眼马赛克（≈实际光学单元分辨率）', fontsize=11)
    ax.axis('off')

    ax = axes[2]
    ax.scatter(uv_c[:, 0] * PAN_W, (1 - uv_c[:, 1]) * PAN_H, s=6, c=drive,
               cmap='inferno', vmin=0, vmax=30, linewidths=0)
    ax.set_title('③ 感光点阵 + 注入电流 30L/(0.02+L)', fontsize=11)
    ax.set_xlim(0, PAN_W); ax.set_ylim(0, PAN_H); ax.invert_yaxis()
    ax.set_facecolor('#111'); ax.axis('off')

    ax = axes[3]
    ax.scatter(uv_c[:, 0] * PAN_W, (1 - uv_c[:, 1]) * PAN_H, s=6, c=lum_agc,
               cmap='viridis', vmin=0, vmax=1, linewidths=0)
    ax.set_title('④ AGC 光适应后（对比拉伸，果蝇实际可见）', fontsize=11)
    ax.set_xlim(0, PAN_W); ax.set_ylim(0, PAN_H); ax.invert_yaxis()
    ax.set_facecolor('#111'); ax.axis('off')

    stats = (f'全景 1920×480 = 左90°+前90°+右90°  '
             f'L眼→左135° / R眼→右135° / 中央重叠10%  '
             f'原始亮度均值 {lum.mean():.3f}  近黑(<0.05): {(lum<0.05).mean()*100:.1f}%  '
             f'AGC后均值 {lum_agc.mean():.3f}  '
             f'→ 结论：原始亮度近全黑，信息主要在运动/对比通道')
    fig.text(0.02, 0.01, stats, fontsize=9.5, color='#333')

    out = OUT_DIR / 'fly_eye_panorama_corrected.png'
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved:', out, flush=True)

if __name__ == '__main__':
    main()
