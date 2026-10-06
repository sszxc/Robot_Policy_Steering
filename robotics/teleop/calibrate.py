"""Fit mapping.CALIB presets (GAIN, OFFSET) from ROM recordings; prints a block to paste into mapping.py.

full   (B): human p2..p98 raw angle -> robot [LO, HI], per joint.
anchor (A): flexion joints keep 0 -> 0 and scale p98 -> HI; roll centred on the open-hand median and scaled so the
            wider side reaches the limit; thumb cmc/yaw stay geometric (gain 1, they are the metacarpal IK).
Usage: python -m robotics.teleop.calibrate [files.npz ...]
"""
import argparse

import numpy as np

from robotics.teleop import mapping
from robotics.teleop.plot_rom import DEFAULT, MIN_CONF, NAMES, ROWS

PCT = (2, 98)
ROLL = [4, 8, 12, 16]


def fit(q, q_open):
    lo, hi = np.percentile(q, PCT, axis=0)
    full_g = (mapping.HI - mapping.LO) / (hi - lo)
    full = full_g, mapping.LO - full_g * lo

    g, o = np.ones(20), np.zeros(20)
    flex = mapping.LO == 0
    g[flex] = mapping.HI[flex] / hi[flex]
    c = np.median(q_open[:, ROLL], axis=0)
    g[ROLL] = mapping.HI[ROLL] / np.maximum(hi[ROLL] - c, c - lo[ROLL])
    o[ROLL] = -g[ROLL] * c
    return {"anchor": (g, o), "full": full}, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=DEFAULT)
    args = ap.parse_args()

    q, ext = [], []
    for f in args.files:
        r = np.load(f)
        keep = r["confidence"] > MIN_CONF
        q.append(np.array([mapping.angles(k) for k in r["kp"][keep].astype(np.float64)]))
        ext.append(r["extended"][keep].all(1))
    q, ext = np.vstack(q), np.concatenate(ext)
    calib, lo, hi = fit(q, q[ext])

    print(f"{len(q)} frames, {ext.sum()} open-hand; open-hand q = median robot target of those frames")
    print(f"{'joint':13s} {'human p2..p98':>14s} {'robot LO..HI':>13s} | " + " | ".join(f"{n:>6s} gain/off/open" for n in calib))
    for j in range(20):
        cells = []
        for g, o in calib.values():
            q_open = np.median(np.clip(g[j] * q[ext, j] + o[j], mapping.LO[j], mapping.HI[j]))
            cells.append(f"{g[j]:5.2f} {o[j]:6.2f} {q_open:5.2f}")
        print(f"{ROWS[j // 4] + ' ' + NAMES[j]:13s} {lo[j]:6.2f}..{hi[j]:5.2f} {mapping.LO[j]:6.2f}..{mapping.HI[j]:5.2f} | "
              + " | ".join(cells))
    fmt = lambda a: "np.array([" + ", ".join(f"{x:.3f}" for x in a) + "])"
    print("\n" + "\n".join(f'    "{n}": ({fmt(g)},\n{" " * (len(n) + 8)}{fmt(o)}),' for n, (g, o) in calib.items()))


if __name__ == "__main__":
    main()
