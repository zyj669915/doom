# -*- coding: utf-8 -*-
"""Luminance-fix rendering (no panorama / no stitching): what the fly REALLY
receives under the current pipeline vs the dual-channel fix, on a normal
90-deg combat frame.

Panels (left-to-right):
  1. original game frame (640x480)
  2. CURRENT pipeline input to the brain: agc(motion(lum))   (v4_agc_motion)
  3. FIXED dual-channel input: 0.5*lum + 0.5*agc(motion(lum))
  4. injected drive under the fix: 30*L_in/(0.02+L_in)
Stats show mean drive / near-black fraction for each channel.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyh.ttf', r'C:\Windows\Fonts\simhei.ttf']:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

from doom.game import Game, retinal_samples
from readout_experiment import agc, motion

ROOT = Path(r'D:\fly\doomfly')
GRAPH = ROOT / 'outputs' / 'doom' / 'malecns_v1' / 'graph.npz'
OUT_DIR = Path(r'D:\fly\reference\fly-vision')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    g = np.load(GRAPH)
    uv = g['uv']
    game = Game(seed=41027, scenario='combat_survival')
    rgb0 = game.pixels()
    game.act({'turn': -6, 'forward': 20, 'attack': 1})
    rgb1 = game.pixels()
    game.close()

    lum0 = retinal_samples(rgb0, uv).astype(np.float32)
    lum1 = retinal_samples(rgb1, uv).astype(np.float32)
    cur_in = agc(motion(lum1, lum0))                    # current v4 pipeline
    fix_in = 0.5 * lum1 + 0.5 * agc(motion(lum1, lum0))  # dual-channel fix
    fix_drive = 30 * fix_in / (0.02 + fix_in)

    def show(ax, vals, cmap, vmin, vmax, title):
        ax.scatter(uv[:, 0] * 640, (1 - uv[:, 1]) * 480, s=6, c=vals,
                   cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 640); ax.set_ylim(0, 480); ax.invert_yaxis()
        ax.set_facecolor('#111'); ax.axis('off')

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6), dpi=110)
    fig.suptitle('亮度修复：果蝇实际收到的视觉信号（90° 视场，不拼接）',
                 fontsize=13, color='#111')
    axes[0].imshow(rgb1); axes[0].set_title('① 原游戏画面 (640×480)', fontsize=11)
    axes[0].axis('off')
    show(axes[1], cur_in, 'viridis', 0, 1, '② 当前管线输入 (AGC+差分)')
    show(axes[2], fix_in, 'viridis', 0, 1, '③ 修复：亮度+差分双通道')
    show(axes[3], fix_drive, 'inferno', 0, 30, '④ 修复后注入电流 30L/(0.02+L)')

    stats = (f'当前管线: 均值 {cur_in.mean():.3f} 近黑<0.1: {(cur_in<0.1).mean()*100:.1f}%  '
             f'亮度响应: 弱(AGC抹平)  |  修复双通道: 均值 {fix_in.mean():.3f} '
             f'近黑<0.1: {(fix_in<0.1).mean()*100:.1f}%  驱动均值 {fix_drive.mean():.1f}')
    fig.text(0.02, 0.01, stats, fontsize=10, color='#333')

    out = OUT_DIR / 'fly_eye_luminance_fix.png'
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved:', out)

if __name__ == '__main__':
    main()
