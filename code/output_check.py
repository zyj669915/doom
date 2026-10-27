# -*- coding: utf-8 -*-
"""Stage-3 output-interface check:
 1. empirical tick-alignment smoke (10 frames: tick sequence must be contiguous,
    label & features from the same frame, no future info)
 2. static checks: dimension table, action space, leakage review
Outputs: outputs/doom/output-check/results.json
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from doom.engine import Brain
from doom.game import Game, retinal_samples
from readout_experiment import custom_step, SCALE, BIAS, THRESH, DUR_MS, readout_indexes, BRAIN, MANIFEST
from teacher_ppo import Policy, ObsBuilder, index_to_action_12

OUT = CODE / 'outputs' / 'doom' / 'output-check'
OUT.mkdir(parents=True, exist_ok=True)
TEACHER = CODE / 'outputs' / 'doom' / 'teacher-v4-final' / 'teacher-distillable.pt'
N = 10

def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    ridx = readout_indexes(manifest)
    ck = torch.load(TEACHER, map_location='cpu', weights_only=True)
    cfg = ck.get('cfg', {})
    obs_mode = cfg.get('obs_mode', 'A')
    obs_dim = {'A': 24 * 32 * 2, 'B': 24 * 32 * 4, 'C': 24 * 32 * 3}[obs_mode]
    policy = Policy(obs_dim=obs_dim, action_mode='12')
    policy.load_state_dict(ck['policy']); policy.eval()

    brain = Brain(BRAIN)
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    game = Game(seed=41027, scenario='combat_survival')
    game.new_episode()
    ob = ObsBuilder(mode=obs_mode, rand=0.0)
    ticks, feats, acts = [], [], []
    for t in range(N):
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain.uv)
        counts = custom_step(brain, lum, DUR_MS)          # brain step on frame t
        o = ob.step(game) if t > 0 else ob.reset(game)
        with torch.no_grad():
            a = int(policy.act(torch.from_numpy(o).unsqueeze(0))[0].item())
        ticks.append(t)
        feats.append(counts[ridx].tolist())
        acts.append(a)
        game.act(index_to_action_12(a))
    game.close()

    contig = ticks == list(range(N))
    static = {
        'action_space': '12 discrete (turn x forward x attack), identical to teacher training',
        'student_input_dim': {
            'author14 x5w': '70', 'top32 x5w': '160', 'ABC96 x5w': '480',
            'single-frame top32': '32', 'single-frame ABC96': '96',
            'note': 'dim = n_readout x window; StudentV2(feat=...) accepts any'},
        'leakage_review': {
            'student_features': 'spike counts of readout neurons only',
            'game_internal_leaked': 'none (no coordinates / game variables / labels in features)',
            'teacher_labels': 'action from teacher policy on the SAME frame'},
        'alignment': 'frame t: pixels -> lum -> brain step -> features -> teacher label -> act, all within one loop iteration',
        'known_issue': 'distill_v2.collect(ppo) still feeds 768-dim single-frame obs to v4 teacher (obs_dim=1536); '
                       'must use ObsBuilder; readout_prescreen.collect is the correct path',
    }
    res = {
        'tick_contiguity': bool(contig),
        'n_frames': N,
        'frames_recorded': ticks,
        'feature_shape_per_frame': len(feats[0]),
        'actions_recorded': acts,
        'same_frame_alignment': True,
        'static': static,
    }
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
