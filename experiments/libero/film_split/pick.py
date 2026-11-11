"""Best axis-grid z (success on the grid inits, then fewer mean success steps) -> OUT.npy [1, dim] for run.py --z-file.
python experiments/libero/film_split/pick.py GRID_DIR OUT.npy {all,gamma,beta}   (split W: axes < dim/2 drive γ)
"""
import sys

import numpy as np

r = np.load(f"{sys.argv[1]}/results.npz")
dim, T = r["Z"].shape[1], r["state"].shape[1]
keep = {"all": range(dim), "gamma": range(dim // 2), "beta": range(dim // 2, dim)}[sys.argv[3]]
assert r["done"].all(), "grid not finished"


def key(k):
    m = r["ep_z"] == k
    s, n = r["success"][m], r["steps"][m]
    return s.mean(), -(n[s].mean() if s.any() else T)


k = max((k for k in range(1, len(r["Z"])) if r["direction"][k] in keep), key=key)
np.save(sys.argv[2], r["Z"][k][None])
print(f"{sys.argv[2]}: row {k}, axis {r['direction'][k]}, scale {r['scale'][k]:+.2f}, grid rate {key(k)[0]:.3f}")
