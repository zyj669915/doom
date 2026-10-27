"""V0: author-original baseline (rule policy on DNp20/DNpe017 with author
params bias=12, thresh=-45, weight NOT scaled) -> closed-loop 600 tics x 3 seeds.
Outputs: phase3-v0/evaluation_baseline.json
"""
import json, sys
from pathlib import Path
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, r'D:\fly\doomfly')

from eval_v2_600 import run_closed_loop, SEEDS, MANIFEST
from distill_fly import readout_indexes

OUT = CODE / 'outputs' / 'doom' / 'phase3-v0'
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
ridx = readout_indexes(manifest)

rows = []
for s in SEEDS:
    r = run_closed_loop('baseline', s, 600, manifest, ridx, None)
    rows.append(r)
    print(json.dumps(r), flush=True)

(OUT / 'evaluation_baseline.json').write_text(json.dumps(rows, indent=1) + '\n')
kills = [r['kills'] for r in rows]
tics = [r['tics'] for r in rows]
print(f'=== V0 author-baseline done | kills={kills} mean={sum(kills)/len(kills):.2f} tics={tics} mean={sum(tics)/len(tics):.0f} ===')
