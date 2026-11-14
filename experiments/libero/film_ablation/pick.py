"""Best grid z over one or more grids (grid success, then fewer mean success steps) -> OUT.npy [1, DIM].
Axes use d16 split names (γ 0-7, β 8-15) whatever each grid's dim, so R′ d8 grids and the d16 grid merge.
python pick.py OUT.npy DIM {gamma,beta} AXES GRID...   AXES: comma list of d16 axes, or - for all
python pick.py OUT.npy DIM raw GRID                     best row as is (2-D combo grid)
Prints the pick key (a{axis}c{scale} or r{row}) on stdout.
"""
import sys

import numpy as np

out, dim, kind, rest = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
axes = None
if kind != "raw":
    axes = None if rest[0] == "-" else {int(a) for a in rest[0].split(",")}
    rest = rest[1:]
cands = []  # (rate, -steps, key, z)
for g in rest:
    r = np.load(f"{g}/results.npz")
    assert r["done"].all(), f"{g} not finished"
    D, T = r["Z"].shape[1], r["state"].shape[1]
    for k in range(1, len(r["Z"])):
        m = r["ep_z"] == k
        s, n = r["success"][m], r["steps"][m]
        score = (s.mean(), -(n[s].mean() if s.any() else T))
        zk = r["Z"][k]
        if kind == "raw":
            cands.append((*score, f"r{k:02d}", zk))
            continue
        a = int(np.abs(zk).argmax())
        c = zk[a] / D**0.5
        a16 = a if a < D // 2 else a - D // 2 + 8
        if (a16 < 8) != (kind == "gamma") or (axes is not None and a16 not in axes):
            continue
        i = a16 if a16 < 8 else a16 - 8 + dim // 2
        assert i < dim and (a16 < 8 or a16 - 8 < dim // 2), f"axis {a16} does not fit dim {dim}"
        z = np.zeros(dim)
        z[i] = c * dim**0.5
        cands.append((*score, f"a{a16:02d}c{c:+.1f}", z))
rate, _, key, z = max(cands, key=lambda x: x[:2])
np.save(out, z[None])
print(f"{out}: {key} grid rate {rate:.3f} ({len(cands)} candidates)", file=sys.stderr)
print(key)
