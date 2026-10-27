# -*- coding: utf-8 -*-
"""Input-side probe (audit section 3): scripted action sequence that saves raw
frames + luminance, then runs 3-segment checks (stationary / self-turn /
enemy-motion) comparing v0_raw vs v4_agc_motion brain response, plus a
photoreceptor-mapping / engineering-assumption check.

Outputs:
  input-probe/probe.npz      rgb_gray(T,480,640) u8, lum_raw, lum_v4, seg_ids
  input-probe/analysis.json  per-segment stats + mapping check
"""
import sys, json, time
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain
from doom.game import Game, retinal_samples
from readout_experiment import (custom_step, readout_indexes, agc, motion,
                                SCALE, BIAS, THRESH, DUR_MS, BRAIN, MANIFEST)

OUT = CODE / 'outputs' / 'doom' / 'input-probe'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 41027
SEG_NAMES = ['stationary', 'turn_left', 'turn_right', 'fwd_fire', 'stationary_fire']
# (tics, action-dict); None = no button (let the episode run on its own)
SCHEDULE = [
    (40, None),
    (40, {'turn': -6, 'forward': 0, 'attack': 0}),
    (40, {'turn': 6, 'forward': 0, 'attack': 0}),
    (40, {'turn': 0, 'forward': 20, 'attack': 1}),
    (40, {'turn': 0, 'forward': 0, 'attack': 1}),
]


def run_raw(seq, name='probe_raw'):
    """v0_raw: luminance straight into the brain (no AGC/motion), same driver."""
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    counts_all = np.zeros((len(seq), brain.n), dtype=np.float32)
    t0 = time.perf_counter()
    for i, l in enumerate(seq):
        counts = custom_step(brain, l, DUR_MS, bias=BIAS, thresh=THRESH)
        counts_all[i] = counts.astype(np.float32)
        if i % 25 == 24:
            print(f'{name} frame {i+1}/{len(seq)} elapsed={time.perf_counter()-t0:.1f}s', flush=True)
    return counts_all, counts_all.sum(0)


def main():
    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    ridx = readout_indexes(manifest)
    game = Game(seed=SEED, scenario='combat_survival')
    gray_l, lum_raw_l, lum_v4_l, seg_l = [], [], [], []
    prev = None
    seg = 0
    for n_tics, act in SCHEDULE:
        for _ in range(n_tics):
            if game.game.is_episode_finished():
                game.new_episode()
                prev = None
            rgb = game.pixels()
            gray = (rgb.astype(np.float32) @ np.asarray([.2126, .7152, .0722], np.float32)).astype(np.uint8)
            lum = retinal_samples(rgb, brain.uv).astype(np.float32)
            l2 = lum.copy() if prev is None else agc(motion(lum, prev))
            gray_l.append(gray); lum_raw_l.append(lum); lum_v4_l.append(l2); seg_l.append(seg)
            prev = lum.copy()
            if act is not None:
                game.act(act)
        seg += 1
    game.close()
    gray = np.stack(gray_l)
    lum_raw = np.stack(lum_raw_l)
    lum_v4 = np.stack(lum_v4_l)
    seg_ids = np.asarray(seg_l, dtype=np.int64)
    print(f'probe collected: frames={len(gray)} lum_raw={lum_raw.shape} gray={gray.shape}', flush=True)
    np.savez_compressed(OUT / 'probe.npz', rgb_gray=gray, lum_raw=lum_raw,
                        lum_v4=lum_v4, seg_ids=seg_ids)

    # --- full-brain sims: v0_raw vs v4_agc_motion ---
    c_raw, tot_raw = run_raw(lum_raw, 'probe_raw')
    c_v4, tot_v4 = sim_full_v4(lum_raw, 'probe_v4')
    r14_raw = c_raw[:, ridx].astype(np.float32)
    r14_v4 = c_v4[:, ridx].astype(np.float32)

    # --- frame-difference motion proxy (large diff = strong scene change) ---
    diff = np.abs(gray[1:].astype(np.int16) - gray[:-1].astype(np.int16)).mean(axis=(1, 2))
    diff = np.concatenate([[0.0], diff])

    rows = {}
    for s, name in enumerate(SEG_NAMES):
        m = seg_ids == s
        n = int(m.sum())
        rows[name] = {
            'frames': n,
            'diff_mean': round(float(diff[m].mean()), 3),
            'readout14_raw_mean': round(float(r14_raw[m].mean()), 1),
            'readout14_raw_std': round(float(r14_raw[m].std()), 1),
            'readout14_v4_mean': round(float(r14_v4[m].mean()), 1),
            'readout14_v4_std': round(float(r14_v4[m].std()), 1),
            'total_spikes_raw_mean': round(float(c_raw[m].sum(1).mean()), 0),
            'total_spikes_v4_mean': round(float(c_v4[m].sum(1).mean()), 0),
        }
    stat = rows['stationary']
    for s in ['turn_left', 'turn_right']:
        rows[s]['ratio_vs_stationary_raw'] = round(rows[s]['readout14_raw_mean'] / (stat['readout14_raw_mean'] + 1e-6), 2)
        rows[s]['ratio_vs_stationary_v4'] = round(rows[s]['readout14_v4_mean'] / (stat['readout14_v4_mean'] + 1e-6), 2)

    # --- photoreceptor mapping & engineering assumptions ---
    check = {
        'uv_shape': list(brain.uv.shape),
        'uv_x_range': [round(float(brain.uv[:, 0].min()), 3), round(float(brain.uv[:, 0].max()), 3)],
        'uv_y_range': [round(float(brain.uv[:, 1].min()), 3), round(float(brain.uv[:, 1].max()), 3)],
        'retina_n': int(len(brain.retina)),
        'lamina_n': int(len(brain.lamina)),
        'drive_formula': '30*L/(0.02+L), max=%.1f @ L=1' % (30.0 / 1.02),
        'r8_color_channel': 'NOT IMPLEMENTED (retinal_samples returns luma only)',
        'engineering_assumptions': {
            'SCALE': SCALE, 'BIAS': BIAS, 'THRESH': THRESH, 'DUR_MS': DUR_MS,
            'note': 'weights*1.5, lamina bias 6mV, thresh -48mV, 50ms/step, fixed luminance formula; no physiology-calibration data used',
        },
    }
    out = {'segments': rows, 'mapping_check': check,
           'overall': {'readout14_raw_mean': round(float(r14_raw.mean()), 1),
                       'readout14_v4_mean': round(float(r14_v4.mean()), 1)}}
    (OUT / 'analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


def sim_full_v4(seq, name='probe_v4'):
    """Replicate readout_experiment.sim_full (v4_agc_motion) to avoid its
    progress.log side effects."""
    from readout_experiment import sim_full
    return sim_full(seq, black=False, name=name)


if __name__ == '__main__':
    main()
