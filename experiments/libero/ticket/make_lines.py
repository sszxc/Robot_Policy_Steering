"""Ticket-search lines for steer_lines.sbatch as a job array (part%a.txt, PAR=2); local queue: ../local.sh.
Parts 0-3: FiLM-z8 (+ held-out z = 0) and noise-z8 per task; parts 4-5: noise-1600, two tasks each.
python experiments/libero/ticket/make_lines.py B OUT_DIR
"""
import sys
from pathlib import Path

B = int(sys.argv[1])
out = Path(sys.argv[2])
R = "/scratch/xzhan742/thesis/results/smolvla/ticket"
ARMS = {"film8": "--inject expert_film --dim 8 --zero",
        "noise8": "--inject noise --dim 8 --box 3 --cem-sigma 1",
        "noise1600": "--inject noise --dim 1600 --box 3 --cem-sigma 1"}


def line(t, arm):
    return f"python -m research.ticket --policy $POLICY --task-id {t} {ARMS[arm]} --batch-size {B} --output-dir {R}/t{t}/{arm}"


out.mkdir(parents=True, exist_ok=True)
parts = [[line(t, "film8"), line(t, "noise8")] for t in (0, 7, 9, 4)]
parts += [[line(t, "noise1600") for t in ts] for ts in ((0, 7), (9, 4))]
for i, p in enumerate(parts):
    (out / f"part{i}.txt").write_text("\n".join(p) + "\n")
print(sorted(p.name for p in out.iterdir()))
