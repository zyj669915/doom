import json
for tag in ['41027', '41028', '41029']:
    p = rf'D:\fly\experiment\code\outputs\doom\teacher-v4-s3r-{tag}\teacher-log.json'
    log = json.load(open(p))
    print(f'--- s{tag} ({len(log)} evals) ---')
    for e in log:
        print(f"  step {e['step']}: kills={e['kills_mean']:.2f}+-{e['kills_std']:.2f} tics={e['tics_mean']:.0f} health={e['health_mean']:.1f} hit/ammo={e['hit_rate_ammo']*100:.0f}%")
