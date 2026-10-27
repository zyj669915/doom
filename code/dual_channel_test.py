# -*- coding: utf-8 -*-
"""Dual-channel front-end test: keep absolute luminance AND motion contrast.

Current v4_agc_motion feeds  agc(motion(lum,prev))  into the brain, which
normalizes global brightness away (luminance response measured at only +6.9%,
and v3_agc drive shows negative response).  Physiological fix: photoreceptor
luminance channel (absolute level, Hill drive) + motion channel (AGC contrast),
like R1-R6 light adaptation plus T4/T5 motion pathways.

Compare on the input_diagnosis S2 (brightness) and S3 (direction) scenes:
  single  : l_in = agc(motion(lum, prev))                 (current)
  dual    : l_in = 0.5*lum + 0.5*agc(motion(lum, prev))   (proposed)

Metrics: brightness 3-class accuracy & response, direction 2-class separability,
read out through top32.
"""
import sys, json
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain
from readout_experiment import custom_step, sep_score, lin_softmax, SCALE, BIAS, THRESH, DUR_MS, BRAIN, agc, motion
from input_diagnosis import build_sequences, to_lum

OUT = CODE / 'outputs' / 'doom' / 'dual-channel'
OUT.mkdir(parents=True, exist_ok=True)
SETS = np.load(CODE / 'outputs' / 'doom' / 'readout-exp' / 'selected_sets.npz')
TOPO32 = SETS['top32_raw']

def run_brain(lum_seq, mode):
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    counts = np.zeros((len(lum_seq), brain.n), np.float32)
    prev = None
    for i, l in enumerate(lum_seq):
        if mode == 'single':
            l_in = agc(motion(l, prev))
        else:
            l_in = 0.5 * l + 0.5 * agc(motion(l, prev))
        prev = l
        counts[i] = custom_step(brain, l_in, DUR_MS, bias=BIAS, thresh=THRESH).astype(np.float32)
    return counts

def main():
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    seqs = build_sequences()
    out = {}
    for mode in ('single', 'dual'):
        row = {}
        for name in ('S2_brightness', 'S3_direction'):
            frames, labels = seqs[name]
            lum = to_lum(frames, brain)
            counts = run_brain(lum, mode)
            F = counts[:, TOPO32].astype(np.float32)
            if name == 'S2_brightness':
                acc, _ = lin_softmax((F - F.mean(0)) / (F.std(0) + 1e-6), labels, ncls=3, iters=600)
                lo, hi = F[labels == 0].mean(), F[labels == 2].mean()
                resp = (hi - lo) / (lo + 1e-6)
                row[name] = {'acc_3class': round(float(acc), 3), 'response': round(float(resp), 3)}
            else:
                acc, cent = sep_score(F, labels)
                row[name] = {'dir_sep_acc': round(float(acc), 3), 'centroid': round(float(cent), 3)}
            print(f'[{mode:6s} {name}] {row[name]}', flush=True)
        out[mode] = row
    (OUT / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print('saved', OUT / 'results.json')

if __name__ == '__main__':
    main()
