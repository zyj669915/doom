# -*- coding: utf-8 -*-
"""现场验证：修复后的干净大脑对"输入画面"是否有可测反应。
合成 6 个画面条件，经两种输入管线喂入全脑仿真，统计神经活动：
  - 当前训练管线 : agc(motion(lum))        (v4, corridor_rl_q 在用)
  - 修复双通道   : 0.5*lum + 0.5*agc(motion) (render_luminance_fix 方案)
每条件 15 帧 × DUR_MS=50。输出总脉冲/受体脉冲/活跃神经元/top512 平均发放。
"""
import sys, json, time
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain  # noqa: E402
from doom.game import retinal_samples  # noqa: E402
from corridor_rl_q import agc, motion, SCALE, BIAS, THRESH, DUR_MS  # noqa: E402

BRAIN_CLEAN = Path(r'D:\fly\experiment\code\outputs\doom\brain-v0-clean\malecns_v1\graph.npz')
OUT = CODE / 'outputs' / 'doom' / 'input-response-demo'
OUT.mkdir(parents=True, exist_ok=True)
W, H = 640, 480


def make_scene(block_x=None, block_y=220, block_s=60, bg_off=0, brightness=1.0):
    yy = np.linspace(0, 1, H)[:, None]
    xx = np.linspace(0, 1, W)[None, :]
    wall = np.clip(0.35 + 0.2 * np.sin(2 * np.pi * (xx + bg_off * 0.02)), 0, 1)
    floor = np.clip(0.5 + 0.25 * np.sin(2 * np.pi * yy * 6), 0, 1)
    scene = np.where(yy > 0.55, floor[0, None] + 0 * xx, wall * (1 - yy[0, None] * 0.2))
    scene = scene.astype(np.float32)
    if block_x is not None:
        x0, x1 = max(0, int(block_x - block_s / 2)), min(W, int(block_x + block_s / 2))
        y0, y1 = max(0, int(block_y - block_s / 2)), min(H, int(block_y + block_s / 2))
        scene[y0:y1, x0:x1] = 0.9
    return np.clip(scene * brightness, 0, 1).astype(np.float32)


def gray_to_rgb(gray):
    return (gray * 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)


def run_condition(brain, frames, pipe, uv):
    """frames: list of (640,480) float32 gray; pipe in {current, dual}; return stats dict."""
    counts_all = []
    prev = None
    for f in frames:
        lum = retinal_samples(gray_to_rgb(f), uv).astype(np.float32)
        if pipe == 'current':
            l2 = agc(motion(lum, prev))
        else:
            l2 = 0.5 * lum + 0.5 * agc(motion(lum, prev))
        prev = lum
        c = brain.step(l2, DUR_MS)
        counts_all.append(c)
    counts = np.stack(counts_all)  # (T, n)
    total = int(counts.sum())
    receptor = int(counts[:, brain.retina].sum())
    active = int((counts > 0).any(axis=0).sum())
    return {'total_spikes': total, 'receptor_spikes': receptor,
            'active_neurons': active}


def main():
    uv = np.load(BRAIN_CLEAN)['uv']
    # 画面条件（各 15 帧，帧间有运动则 prev 参与差分）
    black = [np.zeros((H, W), np.float32) for _ in range(15)]
    still = [make_scene(block_x=320) for _ in range(15)]
    dark = [make_scene(block_x=320, brightness=0.5) for _ in range(15)]
    bright = [make_scene(block_x=320, brightness=1.5) for _ in range(15)]
    left = [make_scene(block_x=320, bg_off=o) for o in range(15)]
    right = [make_scene(block_x=320, bg_off=-o) for o in range(15)]
    conds = {'A_black': black, 'B_still': still, 'C_dark': dark,
             'D_bright': bright, 'E_bg_left': left, 'F_bg_right': right}

    out = {}
    t_start = time.perf_counter()
    for pipe in ('current', 'dual'):
        out[pipe] = {}
        for name, frames in conds.items():
            brain = Brain(BRAIN_CLEAN)
            brain.weight = (brain.weight * SCALE).astype(np.float32)
            r = run_condition(brain, frames, pipe, uv)
            out[pipe][name] = r
            print(f'[{pipe}] {name:12s} total={r["total_spikes"]:7d} '
                  f'receptor={r["receptor_spikes"]:7d} active={r["active_neurons"]:6d}',
                  flush=True)
            del brain
    print(f'elapsed {time.perf_counter() - t_start:.1f}s')

    # 反应判据
    cur, dual = out['current'], out['dual']
    verdict = {
        'black_vs_still_total': (cur['A_black']['total_spikes'], cur['B_still']['total_spikes']),
        'black_vs_still_receptor': (cur['A_black']['receptor_spikes'], cur['B_still']['receptor_spikes']),
        'dark_vs_bright_current': (cur['C_dark']['total_spikes'], cur['D_bright']['total_spikes']),
        'dark_vs_bright_dual': (dual['C_dark']['total_spikes'], dual['D_bright']['total_spikes']),
        'left_vs_right': (cur['E_bg_left']['total_spikes'], cur['F_bg_right']['total_spikes']),
    }
    out['verdict'] = verdict
    json.dump(out, open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
    print('saved:', OUT / 'results.json')


if __name__ == '__main__':
    main()
