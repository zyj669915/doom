"""Read-only: why do DNa02L upstreams fire while DNa02R upstreams stay silent?

Compares the 650 (R) vs 631 (L) presynaptic neurons of the DNa02 pair:
direct lamina/retina inputs, distance to lamina, cell type, net weight.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    g = np.load(ROOT / "outputs/doom/malecns_v1/graph.npz")
    ptr, post, weight, ids = g["ptr"], g["post"], g["weight"], g["ids"]
    superclass = g["superclass"]
    lamina = set(int(x) for x in g["lamina"])
    retina = set(int(x) for x in g["retina"])
    n = len(ids)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    order = np.argsort(post, kind="stable")
    rptr = np.r_[0, np.cumsum(np.bincount(post, minlength=n))].astype(np.int64)
    rpost = pre[order]

    manifest = json.loads((ROOT / "outputs/doom/malecns_v1/manifest.json").read_text())
    dna02 = [r for r in manifest["readouts"] if r["type"] == "DNa02"]
    mn9 = [r for r in manifest["readouts"] if r["type"] == "MN9"]

    def dist_to(source, targets, max_depth=8):
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
        return None

    rows = []
    for r in dna02 + mn9:
        i = r["index"]
        ups = pre[ptr[i]:ptr[i + 1]]
        iw = weight[ptr[i]:ptr[i + 1]]
        lam_direct = int(np.isin(ups, list(lamina)).sum())
        ret_direct = int(np.isin(ups, list(retina)).sum())
        # upstreams whose OWN presynaptic set touches lamina
        lam_2nd = 0
        ret_2nd = 0
        for u in ups:
            uu = pre[ptr[u]:ptr[u + 1]]
            if np.isin(uu, list(lamina)).any():
                lam_2nd += 1
            if np.isin(uu, list(retina)).any():
                ret_2nd += 1
        d_lam = dist_to(i, lamina)
        d_ret = dist_to(i, retina)
        top_sc = {}
        for u in ups:
            k = str(superclass[u])
            top_sc[k] = top_sc.get(k, 0) + 1
        rows.append({
            "id": r["id"], "type": r["type"], "side": r["side"],
            "n_upstream": len(ups), "net_w": round(float(iw.sum()), 1),
            "direct_lamina_up": lam_direct, "direct_retina_up": ret_direct,
            "lamina_2hop_up": lam_2nd, "retina_2hop_up": ret_2nd,
            "dist_lamina": d_lam, "dist_retina": d_ret,
            "top_upstream_superclass": dict(sorted(top_sc.items(), key=lambda kv: -kv[1])[:4]),
        })
        print(f"{r['type']}{r['side']}: up={len(ups):4d} netW={iw.sum():8.1f} "
              f"lam1={lam_direct:3d} ret1={ret_direct:3d} lam2={lam_2nd:4d} ret2={ret_2nd:4d} "
              f"distLam={d_lam} distRet={d_ret}  {sorted(top_sc.items(), key=lambda kv:-kv[1])[:3]}")

    (ROOT / "outputs/doom/dna02-upstream-audit.json").write_text(
        json.dumps(rows, indent=2) + "\n")

if __name__ == "__main__":
    main()
