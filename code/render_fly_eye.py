# -*- coding: utf-8 -*-
"""What the fly actually sees: original Doom frame -> 3,335 photoreceptor
samples (R1-R6) -> injected current 30L/(0.02+L) -> motion-enhanced visual
input (v4_agc_motion) actually driving the connectome.  4 panels.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyh.ttf', r'C:\Windows\Fonts\simhei.ttf']:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

from doom.game import Game, retinal_samples

ROOT = Path(r'D:\fly\doomfly')
GRAPH = ROOT / 'outputs' / 'doom' / 'malecns_v1' / 'graph.npz'
OUT_DIR = Path(r'D:\fly\reference\fly-vision')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def agc(lum, eps=1e-4):
    sd = lum.std()
    if sd < eps:
        return lum.copy()
    return 0.5 + 0.5 * np.tanh((lum - lum.mean()) / (sd + eps) / 1.5)

def motion(lum, prev, k=1.2):
    if prev is None:
        return lum.copy()
    return np.clip(lum + k * (lum - prev), 0, 1)

def main():
    g = np.load(GRAPH)
    uv = g['uv']
    print('retina mapped:', len(uv), 'uv range', uv.min(0), uv.max(0))

    game = Game(seed=41027, scenario='combat_survival')
    rgb0 = game.pixels()
    # a second frame with the scene in motion (teacher turns & fires)
    game.act({'turn': -6, 'forward': 20, 'attack': 1})
    rgb1 = game.pixels()
    game.close()

    lum0 = retinal_samples(rgb0, uv).astype(np.float32)
    lum1 = retinal_samples(rgb1, uv).astype(np.float32)
    drive = 30 * lum0 / (0.02 + lum0)
    motionv = motion(lum1, lum0)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.4), dpi=110)
    fig.suptitle('DOOMFLY: 人类看到的画面 vs 果蝇眼中的世界 (combat_survival, seed 41027)',
                 fontsize=13, color='#111')

    ax = axes[0]; ax.imshow(rgb1); ax.set_title('① 原游戏画面 (640×480)', fontsize=11)
    ax.axis('off')

    ax = axes[1]
    left = uv[:, 0] < 0.5
    for m, c, lab in [(~left, '#8b8b8b', 'R 眼'), (left, '#d0d0d0', 'L 眼')]:
        ax.scatter(uv[m, 0]*rgb1.shape[1], (1-uv[m, 1])*rgb1.shape[0], s=4,
                   c=lum1[m], cmap='gray', vmin=0, vmax=1, linewidths=0)
    ax.set_title(f'② 果蝇感光点阵 (3335点: L眼{left.sum()} / R眼{(~left).sum()})', fontsize=11)
    ax.set_xlim(0, 640); ax.set_ylim(0, 480); ax.invert_yaxis()
    ax.add_patch(Rectangle((0, 0), 640*0.5, 480, fill=False, edgecolor='#1f6feb', lw=1.5, ls='--'))
    ax.add_patch(Rectangle((640*0.5, 0), 640*0.5, 480, fill=False, edgecolor='#238636', lw=1.5, ls='--'))
    ax.text(5, 15, 'L 眼视野', color='#1f6feb', fontsize=9)
    ax.text(640*0.5+5, 15, 'R 眼视野', color='#238636', fontsize=9)
    ax.set_facecolor('#222'); ax.axis('off')

    ax = axes[2]
    ax.scatter(uv[:, 0]*rgb1.shape[1], (1-uv[:, 1])*rgb1.shape[0], s=4, c=drive,
               cmap='inferno', vmin=0, vmax=30, linewidths=0)
    ax.set_title('③ 大脑实际收到的注入电流 30L/(0.02+L)', fontsize=11)
    ax.set_xlim(0, 640); ax.set_ylim(0, 480); ax.invert_yaxis()
    ax.set_facecolor('#111'); ax.axis('off')

    ax = axes[3]
    ax.scatter(uv[:, 0]*rgb1.shape[1], (1-uv[:, 1])*rgb1.shape[0], s=4, c=motionv,
               cmap='viridis', vmin=0, vmax=1, linewidths=0)
    ax.set_title('④ 果蝇运动视觉 (AGC+差分增强信号)', fontsize=11)
    ax.set_xlim(0, 640); ax.set_ylim(0, 480); ax.invert_yaxis()
    ax.set_facecolor('#111'); ax.axis('off')

    stats = (f'亮度均值 {lum1.mean():.3f} (±{lum1.std():.3f})  '
             f'亮点(>0.5): {(lum1>0.5).mean()*100:.1f}%  '
             f'近黑(<0.05): {(lum1<0.05).mean()*100:.1f}%  '
             f'差分帧帧差 {np.abs(motionv-lum0).mean():.3f}')
    fig.text(0.02, 0.01, stats, fontsize=10, color='#333')

    out = OUT_DIR / 'fly_eye_view.png'
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved:', out)

if __name__ == '__main__':
    main()
