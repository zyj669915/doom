# -*- coding: utf-8 -*-
"""Luminance-availability fix: why is the fly 'seeing black' in dark Doom scenes?

Current drive formula  30*L/(0.02+L)  is a saturating Hill curve: L=0.02 -> 15
(half-saturation), L=1 -> 29.4, so 50x luminance difference compresses to ~2x
drive (S2 showed only +6.9% activity for 0.5->1.5 brightness).  Also the raw
linear-luma rendering panel is misleading: the actual pipeline already applies
AGC (v4_agc_motion) before injection.

This script compares drive variants at the photoreceptor level (no full-brain
sim) on 30 real combat frames: mean drive, near-black fraction, brightness
response, dynamic range, entropy.  Best variant(s) then get a full-brain check.

Variants:
  v0  30L/(0.02+L)                 (current)
  v1  30*sqrt(L)/(0.02+sqrt(L))    (dark-region stretch via sqrt)
  v2  30L/(0.005+L)                (lower saturation const -> wider dark range)
  v3  30*agc(L)                    (AGC-normalized, 0..1 -> 0..30)
  v4  30*L**0.6                    (power-law dark lift)
"""
import sys, json
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain
from doom.game import Game, retinal_samples

OUT = CODE / 'outputs' / 'doom' / 'luminance-variants'
OUT.mkdir(parents=True, exist_ok=True)
BRAIN = Path(r'D:\fly\doomfly\outputs\doom\malecns_v1\graph.npz')

def agc(lum, eps=1e-4):
    sd = lum.std()
    if sd < eps:
        return lum.copy()
    return 0.5 + 0.5 * np.tanh((lum - lum.mean()) / (sd + eps) / 1.5)

VARIANTS = {
    'v0_current': lambda L: 30 * L / (0.02 + L),
    'v1_sqrt': lambda L: 30 * np.sqrt(np.clip(L, 0, 1)) / (0.02 + np.sqrt(np.clip(L, 0, 1))),
    'v2_lowk': lambda L: 30 * L / (0.005 + L),
    'v3_agc': lambda L: 30 * agc(L),
    'v4_pow06': lambda L: 30 * np.clip(L, 0, 1) ** 0.6,
}

def main():
    brain = Brain(BRAIN)
    game = Game(seed=41027, scenario='combat_survival')
    game.new_episode()
    lums = []
    for _ in range(30):
        rgb = game.pixels()
        lums.append(retinal_samples(rgb, brain.uv).astype(np.float32))
        game.act({'turn': 0, 'forward': 20, 'attack': 0})
    game.close()
    L = np.stack(lums)
    print('collected', L.shape, 'lum mean', L.mean().round(4))

    # dark / bright sub-sets (by frame mean luminance) for response metric
    fm = L.mean(1)
    dark = L[fm <= np.percentile(fm, 40)]
    bright = L[fm >= np.percentile(fm, 60)]
    rows = {}
    for name, fn in VARIANTS.items():
        D = fn(dark); B = fn(bright)
        drive = fn(L)
        p10, p90 = np.percentile(drive, 10), np.percentile(drive, 90)
        resp = float((B.mean() - D.mean()) / (D.mean() + 1e-6))
        ent = -np.mean(drive * np.log(drive + 1e-9))
        rows[name] = {
            'mean_drive': round(float(drive.mean()), 2),
            'nearblack_frac_drive_lt_8': round(float((drive < 8).mean()), 4),
            'brightness_response_dark2bright': round(resp, 3),
            'dynamic_range_p90_p10': round(float(p90 / (p10 + 1e-6)), 2),
            'entropy': round(float(ent), 2),
        }
        print(f"[{name:10s}] mean={rows[name]['mean_drive']:5.2f} "
              f"nearblack={rows[name]['nearblack_frac_drive_lt_8']:.3f} "
              f"resp={rows[name]['brightness_response_dark2bright']:.3f} "
              f"range={rows[name]['dynamic_range_p90_p10']:.2f} "
              f"ent={rows[name]['entropy']:.2f}")
    (OUT / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')

if __name__ == '__main__':
    main()
