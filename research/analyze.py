"""Closed-loop steering metrics from research.run outputs (results.npz + provenance.json).

Trajectory feature: eef xyz at fixed steps (held after done). Effects are paired with the z=0
episode of the same (init, rep 0); noise effects are z=0 rep j vs rep 0 of the same init.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TIMES = np.arange(20, 201, 20)
GRIP_CLOSED = 0.06  # finger width [m]; open ≈ 0.08


def load(run):
    d = dict(np.load(run / "results.npz"))
    cfg = json.loads((run / "provenance.json").read_text())["config"]
    m = d["done"]
    ep = {k: d[k][m] for k in ("ep_z", "ep_init", "ep_rep", "success", "steps", "state")}
    ep["scale"], ep["dir"] = d["scale"][ep["ep_z"]], d["direction"][ep["ep_z"]]
    ep["w"] = f"{cfg['w']}-s{cfg['w_seed']}" if cfg["w"] in ("random", "split") else "pca"
    ep["task"] = cfg["task_id"]
    return ep


def features(state):
    width = state[..., 6] - state[..., 7]
    closed = width < GRIP_CLOSED
    grasp_t = np.where(closed.any(1), closed.argmax(1), state.shape[1])
    return state[:, TIMES, :3].reshape(len(state), -1), grasp_t


def main_effect_fraction(d):
    """d[K, R, F] paired effects; share of Σ‖d‖² explained by per-k means (null ≈ 1/R)."""
    return d.shape[1] * (d.mean(1) ** 2).sum() / (d ** 2).sum()


def metrics(eps, zero):
    """eps: one (w, scale) group; zero: z=0 episodes of the same task."""
    fz, gz = features(zero["state"])
    inits = np.unique(eps["ep_init"])
    ref = {r: np.flatnonzero((zero["ep_init"] == r) & (zero["ep_rep"] == 0))[0] for r in inits}
    dirs = np.unique(eps["dir"])
    f, g = features(eps["state"])
    D = np.zeros((len(dirs), len(inits), f.shape[1]))
    S = np.zeros((len(dirs), len(inits)))
    for a, k in enumerate(dirs):
        for b, r in enumerate(inits):
            i = np.flatnonzero((eps["dir"] == k) & (eps["ep_init"] == r))[0]
            D[a, b], S[a, b] = f[i] - fz[ref[r]], eps["success"][i]
    reps = np.unique(zero["ep_rep"])[1:]
    N = np.array([[fz[np.flatnonzero((zero["ep_init"] == r) & (zero["ep_rep"] == j))[0]] - fz[ref[r]]
                   for r in inits] for j in reps])
    Sz = np.array([[zero["success"][(zero["ep_init"] == r) & (zero["ep_rep"] == j)][0]
                    for r in inits] for j in np.unique(zero["ep_rep"])])
    K = min(len(dirs), len(Sz))
    p = S.mean()
    return {
        "n_dirs": len(dirs), "n_inits": len(inits),
        "success": p, "success_zero": Sz.mean(),
        "effect_rms_cm": 100 * np.sqrt((D ** 2).mean() * 3),  # per-timepoint eef displacement
        "noise_rms_cm": 100 * np.sqrt((N ** 2).mean() * 3) if len(N) else np.nan,
        "main_effect_frac": main_effect_fraction(D),
        "main_effect_frac_noise": main_effect_fraction(N) if len(N) else np.nan,
        "null_frac": 1 / len(inits),
        "best_of_k": S[:K].max(0).mean(), "best_of_k_zero": Sz[:K].max(0).mean(), "k": K,
        # var of per-direction success over binomial var; >1 means some z are reliably better/worse
        "overdispersion": S.mean(1).var() / max(p * (1 - p) / len(inits), 1e-9),
        "grasp_t_shift": float(np.nanmean(np.abs(g.astype(float) - gz[[ref[r] for r in eps["ep_init"]]]))),
        "steps": eps["steps"].mean(),
        "_D": D, "_S": S,
    }


def antisymmetry(groups):
    """cos between mean effects of +c·u_k and −c·u_k per direction (−1 = linear, odd response)."""
    out = {}
    for (w, c), m in groups.items():
        if c > 0 and (w, -c) in groups:
            a, b = m["_D"].mean(1), groups[(w, -c)]["_D"].mean(1)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            out[(w, c)] = float(cos.mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    runs = [load(r) for r in args.runs]
    cat = lambda key, ms: np.concatenate([r[key][m] for r, m in ms])
    keys = ("ep_z", "ep_init", "ep_rep", "success", "steps", "state", "scale", "dir")
    out = {}
    for task in sorted({r["task"] for r in runs}):
        rt = [r for r in runs if r["task"] == task]
        zero = {k: cat(k, [(r, r["ep_z"] == 0) for r in rt]) for k in keys}
        groups = {}
        for w in sorted({r["w"] for r in rt}):
            rw = [r for r in rt if r["w"] == w]
            for c in sorted({c for r in rw for c in r["scale"]} - {0.0}):
                eps = {k: cat(k, [(r, r["scale"] == c) for r in rw]) for k in keys}
                n_dirs, n_inits = len(np.unique(eps["dir"])), len(np.unique(zero["ep_init"]))
                if len(eps["dir"]) < n_dirs * n_inits:
                    print(f"skip {w} {c:+.2f}: incomplete ({len(eps['dir'])}/{n_dirs * n_inits})")
                    continue
                groups[(w, c)] = metrics(eps, zero)
        anti = antisymmetry(groups)
        print(f"task {task}: z=0 success {zero['success'].mean():.2f} ({len(zero['success'])} eps)")
        print(f"{'w':>7} {'scale':>6} {'succ':>5} {'eff cm':>7} {'noise':>6} {'mainfx':>6} {'null':>5}"
              f" {'bestK':>5} {'bK z0':>5} {'overd':>5} {'grasp':>5} {'anti':>5}")
        for (w, c), m in groups.items():
            print(f"{w:>7} {c:>+6.2f} {m['success']:5.2f} {m['effect_rms_cm']:7.2f} {m['noise_rms_cm']:6.2f}"
                  f" {m['main_effect_frac']:6.2f} {m['main_effect_frac_noise']:5.2f} {m['best_of_k']:5.2f}"
                  f" {m['best_of_k_zero']:5.2f} {m['overdispersion']:5.2f} {m['grasp_t_shift']:5.1f}"
                  f" {anti.get((w, c), np.nan):5.2f}")
        out[task] = {f"{w}:{c:+.2f}": {k: float(v) for k, v in m.items() if not k.startswith("_")}
                     | {"antisymmetry": anti.get((w, c))} for (w, c), m in groups.items()}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
