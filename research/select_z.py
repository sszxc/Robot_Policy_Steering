"""Pick z from a research.run grid: top-k (dir, scale) by grid success, ties broken by the mean
success of the neighboring scales on the same ray. Writes held-out z rows and line-scan directions.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path, help="research.run output dir")
    parser.add_argument("--top", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True, help="prefix: <out>-val.npy, <out>-line.npy, <out>.json")
    args = parser.parse_args()
    d = np.load(args.grid / "results.npz")
    m = d["done"]
    rows = np.unique(d["ep_z"][m])
    succ = {k: d["success"][m][d["ep_z"][m] == k].mean() for k in rows if k > 0}
    scale, direction = d["scale"], d["direction"]

    def neighbor(k):
        same = sorted((scale[j], j) for j in succ if direction[j] == direction[k] and np.sign(scale[j]) == np.sign(scale[k]))
        i = [j for _, j in same].index(k)
        near = [same[n][1] for n in (i - 1, i + 1) if 0 <= n < len(same)]
        return np.mean([succ[j] for j in near]) if near else 0.0

    ranked = sorted(succ, key=lambda k: (succ[k], neighbor(k)), reverse=True)[:args.top]
    picks = [{"row": int(k), "dir": int(direction[k]), "scale": float(scale[k]),
              "grid_success": float(succ[k]), "neighbor": float(neighbor(k))} for k in ranked]
    U = np.stack([d["Z"][direction == k][0] / scale[direction == k][0] for k in range(direction.max() + 1)])
    best = picks[0]["dir"]
    other = 0 if best != 0 else 1  # an unselected reference ray
    np.save(f"{args.out}-val.npy", d["Z"][ranked])
    np.save(f"{args.out}-line.npy", U[[best, other]])
    Path(f"{args.out}.json").write_text(json.dumps({"grid": str(args.grid), "picks": picks,
                                                     "line_dirs": [best, other]}, indent=1) + "\n")
    print(json.dumps(picks), "line dirs", [best, other])


if __name__ == "__main__":
    main()
