"""Audit the input side of the game readouts (biological vs BCI) in the native graph.

Answers: what drives the readout neurons, is there an excitatory path from the
retina/lamina, and how far away is it? Read-only analysis of graph.npz.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"
MANIFEST = ROOT / "outputs/doom/malecns_v1/manifest.json"
OUT = ROOT / "outputs/doom/readout-inputs-audit.json"

g = np.load(GRAPH)
ptr, post, weight, ids = g["ptr"], g["post"], g["weight"], g["ids"]
superclass = g["superclass"]
retina = set(int(x) for x in g["retina"])
lamina = set(int(x) for x in g["lamina"])
n = len(ids)

# Reconstruct edge sources (CSR is sorted by pre), then build reverse CSR.
pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(ptr))
order = np.argsort(post, kind="stable")
rptr = np.r_[0, np.cumsum(np.bincount(post, minlength=n))].astype(np.int64)
rpost = pre[order]
rweight = weight[order]

manifest = json.loads(MANIFEST.read_text())
readouts = manifest["readouts"]

def reverse_bfs_distances(source, targets, max_depth=10):
    """Min hops from source (reverse direction) to any target; None if > max_depth."""
    visited = np.zeros(n, dtype=bool)
    frontier = [source]
    visited[source] = True
    for depth in range(1, max_depth + 1):
        nxt = []
        for u in frontier:
            for e in range(rptr[u], rptr[u + 1]):
                v = int(rpost[e])
                if not visited[v]:
                    visited[v] = True
                    if v in targets:
                        return depth
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return None

rows = []
for r in readouts:
    i = r["index"]
    s, e = ptr[i], ptr[i + 1]
    iw = weight[s:e]
    up = pre[s:e]
    pos = float(iw[iw > 0].sum())
    neg = float(iw[iw < 0].sum())
    up_sc = {}
    for u in up:
        k = str(superclass[u])
        up_sc[k] = up_sc.get(k, 0) + 1
    top_sc = sorted(up_sc.items(), key=lambda kv: -kv[1])[:5]
    rows.append({
        "id": r["id"], "type": r["type"], "side": r["side"], "index": i,
        "in_edges": int(e - s), "excitatory_w": round(pos, 2), "inhibitory_w": round(neg, 2),
        "net_w": round(pos + neg, 2),
        "retina_in_direct": int(np.isin(up, list(retina)).sum()),
        "lamina_in_direct": int(np.isin(up, list(lamina)).sum()),
        "dist_retina": reverse_bfs_distances(i, retina),
        "dist_lamina": reverse_bfs_distances(i, lamina),
        "top_upstream": top_sc,
    })

summary = {"graph": str(GRAPH), "neurons": n, "readouts": rows}
OUT.write_text(json.dumps(summary, indent=2) + "\n")

print(f"{'type':<9}{'side':<5}{'in':>6}{'excW':>9}{'inhW':>9}{'netW':>9}{'distR':>6}{'distL':>6}  top-upstream")
for r in rows:
    tops = ", ".join(f"{k}:{v}" for k, v in r["top_upstream"][:3])
    print(f"{r['type']:<9}{r['side']:<5}{r['in_edges']:>6}{r['excitatory_w']:>9.1f}{r['inhibitory_w']:>9.1f}{r['net_w']:>9.1f}"
          f"{str(r['dist_retina']):>6}{str(r['dist_lamina']):>6}  {tops}")
print("report ->", OUT)
