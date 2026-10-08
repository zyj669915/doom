"""Record closed-loop gameplay GIFs for a config (baseline/random/distilled)."""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from doom.engine import Brain
from doom.game import Game, retinal_samples
from teacher_ppo import index_to_action
from distill_fly import custom_step, readout_indexes, Student, SCALE, BIAS, THRESH, TICS, DUR_MS, SEEDS

OUT = ROOT/'outputs'/'doom'/'distill-v1'

def frames_for(config, seed, max_tics=210, step=2):
    manifest = json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
    ridx = readout_indexes(manifest)
    brain = Brain(ROOT/'outputs/doom/malecns_v1/graph.npz')
    if config != 'baseline':
        brain.weight = (brain.weight * SCALE).astype(np.float32)
        bias, thresh = BIAS, THRESH
    else:
        bias, thresh = 12.0, -45.0
    game = Game(seed=seed, scenario='combat_survival')
    frames = []
    if config in ('distilled', 'random'):
        st = Student()
        if config == 'distilled':
            ck = torch.load(OUT/'student.pt', map_location='cpu', weights_only=False)
            st.load_state_dict(ck['student'])
            mu, sd = ck['mu'], ck['sd']
        else:
            mu, sd = np.zeros(14), np.ones(14)
        st.mu, st.sd = mu, sd
        st.eval()
    for t in range(max_tics):
        if game.game.is_episode_finished(): break
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain.uv)
        counts = custom_step(brain, lum, DUR_MS, bias=bias, thresh=thresh)
        if config == 'baseline':
            r20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNp20' and r.get('side')=='R'])
            l20 = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNp20' and r.get('side')=='L'])
            dnpe = sum(int(counts[i]) for i in [r['index'] for r in manifest['readouts'] if r['type']=='DNpe017'])
            act = {'turn': float(np.clip((r20-l20)*0.12,-6,6)),
                   'forward': float(np.clip(dnpe*0.4,0,20)), 'attack': dnpe>0}
        else:
            f = (counts[ridx].astype(np.float32) - mu) / sd
            with torch.no_grad():
                ai = int(st(torch.from_numpy(f).unsqueeze(0)).argmax(1).item())
            act = index_to_action(ai)
        game.act(act)
        if t % step == 0:
            frames.append(rgb)
    game.close()
    return frames

def save_gif(frames, path, scale=0.5, fps=10):
    from PIL import Image
    small = [Image.fromarray(f).resize((int(f.shape[1]*scale), int(f.shape[0]*scale))) for f in frames]
    small[0].save(path, save_all=True, append_images=small[1:], duration=1000//fps, loop=0)
    print('wrote', path, len(frames), 'frames')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='distilled', choices=['distilled','baseline','random'])
    ap.add_argument('--seed', type=int, default=41029)
    args = ap.parse_args()
    frames = frames_for(args.config, args.seed)
    save_gif(frames, OUT/f'{args.config}_seed{args.seed}.gif')
