"""libero_10 film_ablation lines for platforms/hpc/slurm/steer_lines.sbatch (PAR=2), one A100 job per file.
grid%a.txt: per task the d16g/d16b grids, then 6 jobs of two layer variants (each line: grid, then its held-out).
dheld%a.txt (submit after the d16 grids): d-subset held-outs and the 2-D combo of one task.
python experiments/libero/film_ablation/make_lines.py TASKS B OUT_DIR   e.g. 2,6,8,9 30 experiments/libero/film_ablation/lines
"""
import sys
from pathlib import Path

tasks, B, out = [int(t) for t in sys.argv[1].split(",")], int(sys.argv[2]), Path(sys.argv[3])
G = "/scratch/xzhan742/thesis/results/smolvla/film_ablation_10"
PAIRS = [("s1", "s2"), ("s16", "f4"), ("m4", "l4"), ("f1", "f2"), ("m1", "m2"), ("l1", "l2")]


def line(t, *units):
    return f"G={G} bash experiments/libero/film_ablation/sol.sh libero_10 {t} {B} {' '.join(units)}"


grid = [[line(t, "grid:d16g"), line(t, "grid:d16b")] for t in tasks]
grid += [[line(t, f"grid:{a}", f"held:{a}"), line(t, f"grid:{b}", f"held:{b}")] for t in tasks for a, b in PAIRS]
dheld = [[line(t, "held:d"), line(t, "combo")] for t in tasks]
out.mkdir(parents=True, exist_ok=True)
for name, parts in (("grid", grid), ("dheld", dheld)):
    for i, p in enumerate(parts):
        (out / f"{name}{i}.txt").write_text("\n".join(p) + "\n")
print(f"grid0-{len(grid) - 1}, dheld0-{len(dheld) - 1} in {out}")
