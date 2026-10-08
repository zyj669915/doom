"""Longer-horizon re-evaluation: baseline vs distilled vs random, 600 tics (10s).
Also probes enemy color detection for the rule teacher v2.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from doom.engine import Brain
from doom.game import Game, retinal_samples
from teacher_ppo import index_to_action
from distill_fly import custom_step, readout_indexes, Student, SCALE, BIAS, THRESH, DUR_MS

OUT = ROOT/'outputs'/'doom'/'distill-v1'

def evaluate_config(name, seed, tics=600):
    manifest = json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
    ridx = readout_indexes(manifest)
    brain = Brain(ROOT/'outputs/doom/malecns_v1/graph.npz')
    if name != 'baseline':
        brain.weight = (brain.weight * SCALE).astype(np.float32)
        bias, thresh = BIAS, THRESH
    else:
        bias, thresh = 12.0, -45.0
    game = Game(seed=seed, scenario='combat_survival')
    if name in ('distilled', 'random'):
        st = Student()
        if name == 'distilled':
            ck = torch.load(OUT/'student.pt', map_location='cpu', weights_only=False)
            st.load_state_dict(ck['student'])
            st.mu, st.sd = ck['mu'].astype(np.float32), ck['sd'].astype(np.float32)
        else:
            st.mu, st.sd = np.zeros(14, np.float32), np.ones(14, np.float32)
        st.eval()
    t0 = time.perf_counter()
    for t in range(tics):
        if game.game.is_episode_finished(): break
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain.uv)
        counts = custom_step(brain, lum, DUR_MS, bias=bias, thresh=thresh)
        if name == 'baseline':
            r20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNp20' and r.get('side')=='R'])
            l20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNp20' and r.get('side')=='L'])
            dnpe = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNpe017'])
            act = {'turn': float(np.clip((r20-l20)*0.12,-6,6)),
                   'forward': float(np.clip(dnpe*0.4,0,20)), 'attack': dnpe>0}
        else:
            f = (counts[ridx].astype(np.float32) - st.mu) / st.sd
            with torch.no_grad():
                ai = int(st(torch.from_numpy(f).unsqueeze(0)).argmax(1).item())
            act = index_to_action(ai)
        game.act(act)
        if t == tics // 2 and name == 'baseline':
            pass
    import vizdoom
    health = int(game.game.get_game_variable(vizdoom.GameVariable.HEALTH))
    kills = int(game.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
    t_used = t + 1
    game.close()
    print(f'[{name:9s}] seed={seed} tics={t_used} kills={kills} health={health} ({time.perf_counter()-t0:.0f}s)', flush=True)
    return {'config': name, 'seed': seed, 'tics': t_used, 'kills': kills, 'health': health}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tics', type=int, default=600)
    ap.add_argument('--configs', default='baseline,distilled,random')
    args = ap.parse_args()
    seeds = [41027, 41028, 41029]
    rows = []
    for name in args.configs.split(','):
        for s in seeds:
            rows.append(evaluate_config(name, s, args.tics))
    json.dump(rows, open(OUT/f'longeval_t{args.tics}.json','w'), indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in rows: agg[r['config']].append(r)
    print('\n=== summary (600 tics / 10s horizon) ===')
    for cfg, rs in agg.items():
        print(f'{cfg:9s} kills={np.mean([r["kills"] for r in rs]):.2f} tics={np.mean([r["tics"] for r in rs]):.0f} health={np.mean([r["health"] for r in rs]):.1f}')

if __name__ == '__main__':
    main()
