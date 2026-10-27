# -*- coding: utf-8 -*-
"""Stage-1 input-interface diagnosis: 6 synthetic visual scenes fed to the fly
visual pathway (retinal sampling -> v4_agc_motion -> full-brain sim), read out
through the selected top32 readout set.

Scenes (each a 640x480 synthetic gray canvas, 25-40 frames):
  S1 stationary  : fixed scene -> baseline stability
  S2 brightness  : global luminance x0.5/1.0/1.5 -> luminance response
  S3 direction   : enemy block moves left vs right -> directional separability
  S4 approach    : enemy block grows (far -> near) -> looming response
  S5 self-turn   : background pattern shifts (simulated own rotation) -> self-motion pollution
  S6 mixed motion: background shifts + enemy moves extra -> target extraction

Outputs: outputs/doom/input-diagnosis/results.json + counts.npz
"""
import sys, json, time
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain
from doom.game import retinal_samples
from readout_experiment import custom_step, sim_full, sep_score, lin_softmax, SCALE, BIAS, THRESH, DUR_MS

OUT = CODE / 'outputs' / 'doom' / 'input-diagnosis'
OUT.mkdir(parents=True, exist_ok=True)
BRAIN = Path(r'D:\fly\doomfly\outputs\doom\malecns_v1\graph.npz')
SETS = np.load(CODE / 'outputs' / 'doom' / 'readout-exp' / 'selected_sets.npz')
TOPO32 = SETS['top32_raw']

W, H = 640, 480

def make_scene(block_x=None, block_y=240, block_s=60, bg_off=0, brightness=1.0):
    """640x480 gray canvas: vertical gradient wall + horizontal stripes floor + bright block."""
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

def build_sequences():
    """Return list of (name, [frames...], labels?) with each frame a 640x480 gray canvas."""
    seqs = {}
    # S1 stationary: enemy center, 25 identical frames
    f0 = make_scene(block_x=320, block_y=220, block_s=60)
    seqs['S1_stationary'] = ([f0.copy() for _ in range(25)], None)
    # S2 brightness: same scene x0.5 / x1.0 / x1.5, 10 frames each
    fbase = make_scene(block_x=320, block_y=220, block_s=60)
    fr = []
    for b in (0.5, 1.0, 1.5):
        fr += [np.clip(fbase * b, 0, 1).astype(np.float32) for _ in range(10)]
    seqs['S2_brightness'] = (fr, np.repeat([0, 1, 2], 10))
    # S3 direction: enemy moves left (500->100) then right (100->500)
    xs_l = np.linspace(500, 100, 20); xs_r = np.linspace(100, 500, 20)
    fr = [make_scene(block_x=float(x), block_y=220, block_s=50) for x in xs_l]
    fr += [make_scene(block_x=float(x), block_y=220, block_s=50) for x in xs_r]
    seqs['S3_direction'] = (fr, np.repeat([0, 1], 20))
    # S4 approach: block grows 20 -> 140
    fr = [make_scene(block_x=320, block_y=220, block_s=float(s)) for s in np.linspace(20, 140, 20)]
    seqs['S4_approach'] = (fr, None)
    # S5 self-turn: background shifts left 4px/frame, enemy static
    fr = [make_scene(block_x=320, block_y=220, block_s=50, bg_off=float(o)) for o in range(25)]
    seqs['S5_selfturn'] = (fr, None)
    # S6 mixed: background shifts + enemy moves opposite (target extraction)
    fr = [make_scene(block_x=float(320 + 8 * o), block_y=220, block_s=50, bg_off=float(o)) for o in range(25)]
    seqs['S6_mixed'] = (fr, None)
    return seqs

def to_lum(frames, brain):
    """640x480 gray -> retinal sampling -> (T, 3335) luminance."""
    lum = np.stack([retinal_samples((f * 255).astype(np.uint8)[:, :, None].repeat(3, axis=2), brain.uv)
                    for f in frames]).astype(np.float32)
    return lum

def main():
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    seqs = build_sequences()
    results = {}
    all_counts = {}
    t_start = time.perf_counter()
    for name, (frames, labels) in seqs.items():
        lum = to_lum(frames, brain)
        counts, _ = sim_full(lum, name=name)
        F = counts[:, TOPO32].astype(np.float32)
        all_counts[name] = counts
        r = {'frames': len(frames)}
        if name == 'S1_stationary':
            r['mean_spikes'] = round(float(F.mean()), 2)
            r['std_spikes'] = round(float(F.std()), 2)
            r['stable'] = bool(F.std() < F.mean() * 0.2 + 1e-6)
        elif name == 'S2_brightness':
            lo, hi = F[labels == 0].mean(), F[labels == 2].mean()
            r['resp_bright'] = round(float((hi - lo) / (lo + 1e-6)), 3)
            acc, _ = lin_softmax((F - F.mean(0)) / (F.std(0) + 1e-6), labels, ncls=3, iters=600)
            r['bright_class_acc'] = round(float(acc), 3)
        elif name == 'S3_direction':
            acc, cent = sep_score(F, labels)
            r['dir_sep_acc'] = round(float(acc), 3)
            r['dir_centroid'] = round(float(cent), 3)
        elif name == 'S4_approach':
            sizes = np.linspace(20, 140, 20)
            act = F.mean(1)
            r['looming_corr'] = round(float(np.corrcoef(sizes, act)[0, 1]), 3)
        elif name == 'S5_selfturn':
            r['mean_spikes'] = round(float(F.mean()), 2)
            r['self_pollution'] = round(float(F.mean() / (results['S1_stationary']['mean_spikes'] + 1e-6)), 3)
        elif name == 'S6_mixed':
            # enemy moves 8px/frame + bg 1unit/frame; S5 was bg-only
            r['mean_spikes'] = round(float(F.mean()), 2)
            r['target_gain_vs_selfturn'] = round(float(F.mean() / (results['S5_selfturn']['mean_spikes'] + 1e-6)), 3)
        results[name] = r
        print(f'[{name}] {json.dumps(r, ensure_ascii=False)}', flush=True)
        np.save(OUT / f'counts-{name}.npy', counts)
    json.dump({'results': results, 'top32_n': int(len(TOPO32))},
              open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
    print(f'total elapsed {time.perf_counter() - t_start:.1f}s')

if __name__ == '__main__':
    main()
