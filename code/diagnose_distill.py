"""Diagnose the distill link: where does information get lost?
- behavior-clone accuracy of student vs teacher on UNSEEN frames
- per-readout spike activity in live gameplay (which neurons actually carry signal)
- action distribution: teacher vs student
- alignment check: features are from frame t, teacher label from frame t-1 (one-frame lag)
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from doom.engine import Brain
from doom.game import Game, retinal_samples
from teacher_ppo import Policy, index_to_action, obs_vec
from distill_fly import custom_step, readout_indexes, Student, SCALE, BIAS, THRESH, DUR_MS

OUT = ROOT/'outputs'/'doom'/'distill-v1'
TEACHER = OUT.parent/'teacher-v1'

def run_diag(seed, max_tics=210):
    manifest = json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
    ridx = readout_indexes(manifest)
    brain = Brain(ROOT/'outputs/doom/malecns_v1/graph.npz')
    brain.weight = (brain.weight * SCALE).astype(np.float32)
    game = Game(seed=seed, scenario='combat_survival')

    teacher = Policy()
    ck = torch.load(TEACHER/'teacher.pt', map_location='cpu', weights_only=True)
    teacher.load_state_dict(ck['policy']); teacher.eval()

    st = Student()
    ck2 = torch.load(OUT/'student.pt', map_location='cpu', weights_only=False)
    st.load_state_dict(ck2['student']); st.mu, st.sd = ck2['mu'], ck2['sd']; st.eval()

    rows = []
    o = obs_vec(game)
    for t in range(max_tics):
        if game.game.is_episode_finished(): break
        rgb = game.pixels()
        lum = retinal_samples(rgb, brain.uv)
        counts = custom_step(brain, lum, DUR_MS)
        f = (counts[ridx].astype(np.float32) - st.mu) / st.sd
        with torch.no_grad():
            xo = torch.from_numpy(o).unsqueeze(0)
            xt = torch.from_numpy(f).unsqueeze(0)
            # teacher sampled action (what it did) + argmax (its preferred action)
            logits_t, _ = teacher.forward(xo)
            a_sampled = int(torch.distributions.Categorical(logits=logits_t).sample().item())
            a_targ = int(logits_t.argmax(1).item())
            logits_s = st(xt)
            a_stud = int(logits_s.argmax(1).item())
        rows.append({'t': t, 't_sampled': a_sampled, 't_argmax': a_targ,
                     's_argmax': a_stud, 'feat': counts[ridx].copy()})
        game.act(index_to_action(a_sampled))
        o = obs_vec(game)
    game.close()
    return rows

def main():
    seeds = [41031, 41032, 41033]
    all_rows = []
    for s in seeds:
        rows = run_diag(s)
        all_rows.extend(rows)
        print(f'seed {s}: {len(rows)} frames', flush=True)
    n = len(all_rows)
    ts = np.array([r['t_sampled'] for r in all_rows])
    ta = np.array([r['t_argmax'] for r in all_rows])
    sa = np.array([r['s_argmax'] for r in all_rows])
    F = np.stack([r['feat'] for r in all_rows]).astype(np.float32)
    # accuracy: student argmax vs teacher sampled / teacher argmax
    acc_samp = (sa == ts).mean()
    acc_targ = (sa == ta).mean()
    print(f'\nframes={n}')
    print(f'student vs teacher-sampled action: {acc_samp:.3f}')
    print(f'student vs teacher-argmax action : {acc_targ:.3f}  (upper bound ~ teacher determinism)')
    # teacher's own sampled vs argmax consistency
    self_cons = (ts == ta).mean()
    print(f'teacher sampled vs teacher argmax : {self_cons:.3f}')
    # action distributions
    print('\nteacher-sampled dist:', np.bincount(ts, minlength=12))
    print('teacher-argmax dist :', np.bincount(ta, minlength=12))
    print('student-argmax dist :', np.bincount(sa, minlength=12))
    # per-readout activity
    manifest = json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
    ridx = readout_indexes(manifest)
    names = [f"{r['type']}_{r.get('side','')}" for r in manifest['readouts']]
    act = (F > 0).mean(0)
    print('\nper-readout firing rate (frac of frames with >=1 spike):')
    for nm, idx, fr in sorted(zip(names, ridx, act), key=lambda z: -z[2]):
        print(f'  {nm:14s} idx={idx:7d}  {fr:.3f}')
    # confusion: for each teacher argmax action, what does student predict (conditional)
    print('\nstudent argmax given teacher argmax (rows=teacher, cols=student):')
    for a in range(12):
        if (ta == a).sum() > 0:
            cond = np.bincount(sa[ta == a], minlength=12)
            top = np.argsort(cond)[-3:][::-1]
            print(f'  teacher {a:2d} (n={(ta==a).sum():4d}): student-> {[(int(i), int(cond[i])) for i in top]}')
    json.dump({'n': n, 'acc_sampled': float(acc_samp), 'acc_argmax': float(acc_targ),
               'self_cons': float(self_cons), 'firing': {nm: float(fr) for nm, fr in zip(names, act)}},
              open(OUT/'diagnose.json','w'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main()
