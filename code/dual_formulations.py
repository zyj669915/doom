# -*- coding: utf-8 -*-
"""Iterate dual-channel formulations on S2 (brightness) + S3 (direction).
Goal: luminance response must not regress vs single-channel (acc>=0.5 on 3-class)
while keeping/improving the direction gain (single=0.625).

Formulations (l_in fed to brain, brain applies 30*x/(0.02+x) internally):
  single : agc(motion(lum, prev))
  A      : 0.5*lum + 0.5*agc(motion)          (tested: dir 0.75, bright 0.167)
  B      : 0.3*lum + 0.7*agc(motion)
  C      : 0.5*hill(lum) + 0.5*agc(motion), hill=L/(0.02+L)
  D      : 0.6*hill(lum) + 0.4*agc(motion)
  E      : hill(lum) + agc(motion)  (both on [0,1]; additive)
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

def hill(L):
    return L / (0.02 + np.clip(L, 0, 1))

def make_l(lum, prev, f):
    if f == 'single':
        return agc(motion(lum, prev))
    if f == 'A':
        return 0.5 * lum + 0.5 * agc(motion(lum, prev))
    if f == 'B':
        return 0.3 * lum + 0.7 * agc(motion(lum, prev))
    if f == 'C':
        return 0.5 * hill(lum) + 0.5 * agc(motion(lum, prev))
    if f == 'D':
        return 0.6 * hill(lum) + 0.4 * agc(motion(lum, prev))
    if f == 'E':
        return np.clip(hill(lum) + agc(motion(lum, prev)), 0, 1)
    raise ValueError(f)

def run_brain(lum_seq, f):
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    counts = np.zeros((len(lum_seq), brain.n), np.float32)
    prev = None
    for i, l in enumerate(lum_seq):
        counts[i] = custom_step(brain, make_l(l, prev, f), DUR_MS, bias=BIAS, thresh=THRESH).astype(np.float32)
        prev = l
    return counts

def main():
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    seqs = build_sequences()
    lum_s2 = to_lum(seqs['S2_brightness'][0], brain); lab_s2 = seqs['S2_brightness'][1]
    lum_s3 = to_lum(seqs['S3_direction'][0], brain); lab_s3 = seqs['S3_direction'][1]
    results = {}
    for f in ('single', 'A', 'B', 'C', 'D', 'E'):
        c2 = run_brain(lum_s2, f); F2 = c2[:, TOPO32]
        acc, _ = lin_softmax((F2 - F2.mean(0)) / (F2.std(0) + 1e-6), lab_s2, ncls=3, iters=600)
        lo, hi = F2[lab_s2 == 0].mean(), F2[lab_s2 == 2].mean()
        resp = float((hi - lo) / (lo + 1e-6))
        c3 = run_brain(lum_s3, f); F3 = c3[:, TOPO32]
        dacc, cent = sep_score(F3, lab_s3)
        results[f] = {'bright_acc': round(float(acc), 3), 'bright_resp': round(resp, 3),
                      'dir_acc': round(float(dacc), 3), 'dir_centroid': round(float(cent), 3)}
        print(f"[{f:6s}] bright_acc={acc:.3f} resp={resp:.3f} | dir_acc={dacc:.3f} cent={cent:.3f}", flush=True)
    (OUT / 'formulations.json').write_text(json.dumps(results, ensure_ascii=False, indent=1) + '\n')
    print('saved formulations.json')

if __name__ == '__main__':
    main()
