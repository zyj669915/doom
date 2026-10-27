"""Stage-4 final eval: 3 candidate teachers x 3 scenarios, unified protocol.
Computes per-checkpoint metrics + cross-scenario kill transfer rate + action entropy.
"""
import json, math
import torch
from pathlib import Path

from teacher_ppo import Policy, EVAL_SEEDS, eval_policy

OBS_DIM = {'A': 24 * 32 * 2, 'B': 24 * 32 * 4, 'C': 24 * 32 * 3}
BASE = Path(r'D:\fly\experiment\code\outputs\doom')
SCENARIOS = ['combat_survival', 'defend_the_center', 'health_gathering']

CANDIDATES = {
    's1_peak122k': BASE / 'teacher-v4-s3r-41027' / 'teacher-122880.pt',
    's2_peak131k': BASE / 'teacher-v4-s3r-41028' / 'teacher-131072.pt',
    's3_peak122k': BASE / 'teacher-v4-s3r-41029' / 'teacher-122880.pt',
}

def action_entropy(dist):
    return -sum(p * math.log(p) for p in dist if p > 0) if dist else float('nan')

results = {}
for name, path in CANDIDATES.items():
    ck = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ck.get('cfg', {})
    obs_mode = cfg.get('obs_mode', 'A')
    act_mode = cfg.get('action_mode', '12')
    policy = Policy(obs_dim=OBS_DIM[obs_mode], action_mode=act_mode)
    policy.load_state_dict(ck['policy'])
    policy.eval()
    row = {}
    for sc in SCENARIOS:
        res = eval_policy(policy, seeds=EVAL_SEEDS, scenario=sc, tics=600, obs_mode=obs_mode)
        m = res['metrics']
        row[sc] = m
        print(f"[{name}] {sc}: kills={m['kills']:.2f}±{m['kills_std']:.2f} tics={m['tics']:.0f} "
              f"health={m['health']:.1f} hit/ammo={m['hit_rate_ammo']*100:.1f}%", flush=True)
    # cross-scenario kill transfer vs combat
    combat_k = row['combat_survival']['kills']
    row['transfer_defend'] = row['defend_the_center']['kills'] / max(1e-6, combat_k)
    row['transfer_health'] = row['health_gathering']['kills'] / max(1e-6, combat_k)
    row['entropy_combat'] = action_entropy(row['combat_survival']['action_dist'])
    results[name] = row
    print(f"[{name}] transfer_defend={row['transfer_defend']:.2f} "
          f"transfer_health={row['transfer_health']:.2f} entropy={row['entropy_combat']:.2f} nats", flush=True)

out = BASE / 'teacher-v4-stage4-eval.json'
json.dump(results, open(out, 'w'), ensure_ascii=False, indent=1)
print('saved', out)
