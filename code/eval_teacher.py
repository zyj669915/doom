"""Unified teacher evaluation (v4): fixed protocol for ANY v4 checkpoint.
Auto-infers obs_dim / action_mode from checkpoint cfg.
Usage:
  python eval_teacher.py --teacher-path <ck.pt> --out <out.json> [--scenario ...] [--seeds ...] [--tics 600]
"""
from pathlib import Path
import json, argparse

from teacher_ppo import Policy, EVAL_SEEDS, eval_policy

OBS_DIM = {'A': 24 * 32 * 2, 'B': 24 * 32 * 4, 'C': 24 * 32 * 3}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-path', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenario', default='combat_survival')
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--tics', type=int, default=600)
    args = ap.parse_args()

    seeds = [int(s) for s in (args.seeds or ','.join(map(str, EVAL_SEEDS))).split(',') if s]
    ck = __import__('torch').load(args.teacher_path, map_location='cpu', weights_only=True)
    cfg = ck.get('cfg', {})
    obs_mode = cfg.get('obs_mode', 'A')
    action_mode = cfg.get('action_mode', '12')
    policy = Policy(obs_dim=OBS_DIM[obs_mode], action_mode=action_mode)
    policy.load_state_dict(ck['policy'])
    policy.eval()

    res = eval_policy(policy, seeds=seeds, scenario=args.scenario, tics=args.tics, obs_mode=obs_mode)
    m = res['metrics']
    m['scenario'] = args.scenario
    m['obs_mode'] = obs_mode
    m['action_mode'] = action_mode
    m['teacher_path'] = args.teacher_path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f"== {args.scenario} | obs={obs_mode} act={action_mode} ==")
    print(f"kills={m['kills']:.2f}±{m['kills_std']:.2f}  tics={m['tics']:.0f}  health={m['health']:.1f} "
          f"health_lost={m['health_lost']:.1f}  hit/ammo={m['hit_rate_ammo']*100:.1f}%  "
          f"hit/frames={m['hit_rate_frames']*100:.1f}%  ammo/kill={m['ammo_per_kill']:.1f}")

if __name__ == '__main__':
    main()
