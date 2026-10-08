"""Why does the frame look fine to humans but 'dark' to the fly?
Compares: raw sRGB pixel values (what we see) vs linear luminance (physical light)
vs what the fly's 3,335 photoreceptor samples capture. Also checks where samples fall.
"""
from pathlib import Path
import numpy as np
import json
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
OUT = Path(r'D:\fly\reference\fly-vision')
OUT.mkdir(parents=True, exist_ok=True)

def linear_luma(pixels):
    p = pixels.astype(np.float32) / 255
    p = np.where(p <= .04045, p/12.92, ((p+.055)/1.055)**2.4)
    return p @ np.asarray([.2126, .7152, .0722], dtype=np.float32)

def main():
    g = np.load(GRAPH)
    uv = g['uv']
    game = Game(seed=41027, scenario='combat_survival')
    rgb = game.pixels().astype(np.float32)
    game.close()

    # --- whole-frame stats ---
    srgb_gray = rgb.mean(axis=2)                      # 0..255 "what we see"
    lin = linear_luma(rgb)                            # 0..1 physical light
    h, w = lin.shape
    # --- fly samples ---
    lum = retinal_samples(rgb, uv)
    x = uv[:, 0]*(w-1); y = uv[:, 1]*(h-1)
    x0 = x.astype(int); y0 = y.astype(int)
    sampled_srgb = srgb_gray[y0, x0]

    # sRGB -> linear of the sampled mean, to compare apples to apples
    def srgb_to_lin(v):  # v in 0..1
        v = np.clip(v/255, 0, 1)
        return np.where(v <= .04045, v/12.92, ((v+.055)/1.055)**2.4)

    stats = {
        'frame_srgb_mean_0_255': float(srgb_gray.mean()),
        'frame_srgb_median_0_255': float(np.median(srgb_gray)),
        'frame_linear_mean_0_1': float(lin.mean()),
        'frame_linear_median_0_1': float(np.median(lin)),
        'sample_srgb_mean_0_255': float(sampled_srgb.mean()),
        'sample_linear_mean_0_1': float(lum.mean()),
        'sample_pct_below_0_05_linear': float((lum < .05).mean()*100),
        'sample_pct_below_0_02_linear': float((lum < .02).mean()*100),
        'sample_pct_above_0_5_linear': float((lum > .5).mean()*100),
        'frame_pct_below_0_05_linear': float((lin < .05).mean()*100),
        'frame_pct_above_0_5_linear': float((lin > .5).mean()*100),
    }
    # what sRGB value corresponds to linear 0.077? (gamma-encode)
    def lin_to_srgb(v):
        v = np.clip(v, 0, 1)
        return np.where(v <= .0031308, v*12.92, 1.055*v**(1/2.4) - .055) * 255
    stats['linear_0_077_equals_srgb_approx'] = float(lin_to_srgb(0.077))
    stats['srgb_128_equals_linear_approx'] = float(srgb_to_lin(128.0))

    with open(OUT/'brightness-stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # --- figure: histograms ---
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=110)
    fig.suptitle('为什么人看着不暗、果蝇却"看不清"：sRGB 感知亮度 vs 物理线性亮度', fontsize=13)
    ax = axes[0]
    ax.imshow(rgb.astype(np.uint8)); ax.set_title('① 原画面（sRGB 编码，人眼感知）', fontsize=11); ax.axis('off')
    ax = axes[1]
    ax.hist(srgb_gray.ravel(), bins=64, color='#79c0ff', alpha=.8, label='全画面 sRGB')
    ax.hist(sampled_srgb.ravel(), bins=64, color='#f0b429', alpha=.7, label='果蝇采样 sRGB')
    ax.axvline(128, color='#ff5d5d', ls='--', lw=1)
    ax.text(132, ax.get_ylim()[1]*.95, '128 = "中等灰"', color='#ff5d5d', fontsize=9)
    ax.set_xlabel('sRGB 像素值 0–255'); ax.set_ylabel('像素数')
    ax.set_title('② sRGB 亮度分布（人眼感知空间）', fontsize=11); ax.legend(fontsize=9)
    ax = axes[2]
    ax.hist(lin.ravel(), bins=64, color='#4ade80', alpha=.8, label='全画面线性亮度')
    ax.hist(lum.ravel(), bins=64, color='#f0b429', alpha=.7, label='果蝇采样线性亮度')
    ax.axvline(0.077, color='#ff5d5d', ls='--', lw=1)
    ax.text(0.082, ax.get_ylim()[1]*.95, '均值 0.077', color='#ff5d5d', fontsize=9)
    ax.set_xlabel('线性亮度 0–1（物理光强）'); ax.set_ylabel('点数')
    ax.set_title('③ 线性亮度分布（物理光强，果蝇感光输入）', fontsize=11); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/'brightness_explained.png', bbox_inches='tight')
    plt.close(fig)
    print('saved:', OUT/'brightness_explained.png')

if __name__ == '__main__':
    main()
