"""Render what the fly actually sees: original Doom frame vs. retinal sampling.

Outputs a side-by-side PNG:
  left  : raw game frame (640x480)
  right : the 3,335 photoreceptor samples as seen by the fly brain
          (bilinear linear-luma samples at R1-R6 UV positions from the connectome).
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

def main():
    g = np.load(GRAPH)
    uv = g['uv']                 # (3335, 2) in [0,1]
    retina = g['retina']         # neuron ids of mapped photoreceptors
    print('retina mapped:', len(uv), 'uv range', uv.min(0), uv.max(0))

    game = Game(seed=41027, scenario='combat_survival')
    rgb = game.pixels()          # 640x480 RGB24
    game.close()

    lum = retinal_samples(rgb, uv)                 # linear luminance [0,1]
    drive = 30 * lum / (0.02 + lum)                # what the brain receives

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), dpi=110)
    fig.suptitle('DOOMFLY: 人类看到的画面 vs 果蝇看到的画面 (combat_survival, seed 41027)',
                 fontsize=13, color='#111')

    ax = axes[0]; ax.imshow(rgb); ax.set_title('① 原游戏画面 (640×480)', fontsize=11)
    ax.axis('off')

    # ② photoreceptor point cloud, colored by luminance
    ax = axes[1]
    # eye side: prepare.py maps L eye to left 60% viewport, R eye to right 40%
    left = uv[:, 0] < 0.5
    ax.scatter(uv[~left, 0]*rgb.shape[1], (1-uv[~left, 1])*rgb.shape[0], s=4,
               c=lum[~left], cmap='gray', vmin=0, vmax=1, linewidths=0)
    ax.scatter(uv[left, 0]*rgb.shape[1], (1-uv[left, 1])*rgb.shape[0], s=4,
               c=lum[left], cmap='gray', vmin=0, vmax=1, linewidths=0, marker='o')
    ax.set_title(f'② 果蝇感光点阵 (3335点: L眼{left.sum()} / R眼{(~left).sum()})', fontsize=11)
    ax.set_xlim(0, 640); ax.set_ylim(0, 480); ax.invert_yaxis()
    ax.add_patch(Rectangle((0,0), 640*0.5, 480, fill=False, edgecolor='#1f6feb', lw=1.5, ls='--'))
    ax.text(5, 15, 'L 眼视野', color='#1f6feb', fontsize=9)
    ax.text(640*0.5+5, 15, 'R 眼视野', color='#238636', fontsize=9)
    ax.add_patch(Rectangle((640*0.5,0), 640*0.5, 480, fill=False, edgecolor='#238636', lw=1.5, ls='--'))
    ax.set_facecolor('#222'); ax.axis('off')

    # ③ what the brain actually receives: nonlinear drive 30L/(0.02+L)
    ax = axes[2]
    ax.scatter(uv[:, 0]*rgb.shape[1], (1-uv[:, 1])*rgb.shape[0], s=4, c=drive,
               cmap='inferno', vmin=0, vmax=30, linewidths=0)
    ax.set_title('③ 大脑实际收到的注入电流 30L/(0.02+L)', fontsize=11)
    ax.set_xlim(0, 640); ax.set_ylim(0, 480); ax.invert_yaxis()
    ax.set_facecolor('#111'); ax.axis('off')

    stats = (f'亮度均值 {lum.mean():.3f} (±{lum.std():.3f})  '
             f'亮点(>0.5): {(lum>0.5).mean()*100:.1f}%  '
             f'零信息(纯黑): {(lum<0.05).mean()*100:.1f}%')
    fig.text(0.02, 0.01, stats, fontsize=10, color='#333')

    out = OUT_DIR / 'fly_eye_view.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('saved:', out)

if __name__ == '__main__':
    main()
