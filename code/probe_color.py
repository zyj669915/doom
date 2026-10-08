"""Probe enemy color on saved frames to design a reliable target detector."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
for t in (10, 60, 110, 160):
    p = ROOT/'outputs'/'doom'/('probe_frame_%03d.png' % t)
    if not p.exists():
        print('missing', p); continue
    rgb = np.asarray(Image.open(p)).astype(np.float32)
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    h, w = r.shape
    # warm-saturated mask: R dominant and bright
    warm = (r > g + 10) & (r > b + 10) & (r > 60)
    sat = (r - b) / (r + 1e-3)
    warm2 = warm & (sat > 0.25)
    n = int(warm2.sum())
    if n > 0:
        ys, xs = np.where(warm2)
        print(f't={t}: warm_sat_px={n} ({n/(h*w)*100:.1f}%)  cx={xs.mean()/w:.2f} cy={ys.mean()/h:.2f}  box_x=[{xs.min()/w:.2f},{xs.max()/w:.2f}] box_y=[{ys.min()/h:.2f},{ys.max()/h:.2f}]')
        # color sample
        yy, xx = np.where(warm2 & (r > 100))
        if len(yy):
            i = np.random.choice(len(yy), 200)
            print(f'   sample mean RGB: {rgb[yy[i],xx[i]].mean(0).round(0)}')
    else:
        print(f't={t}: no warm-saturated pixels')
