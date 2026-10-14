"""Ticket-search lines for steer_lines.sbatch as a job array (part%a.txt, PAR=2), plus the local t2 script.
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


def line(t, arm, root=R, policy="--policy $POLICY ", b=B):
    return f"python -m research.ticket {policy}--task-id {t} {ARMS[arm]} --batch-size {b} --output-dir {root}/t{t}/{arm}"


out.mkdir(parents=True, exist_ok=True)
parts = [[line(t, "film8"), line(t, "noise8")] for t in (0, 7, 9, 4)]
parts += [[line(t, "noise1600") for t in ts] for ts in ((0, 7), (9, 4))]
for i, p in enumerate(parts):
    (out / f"part{i}.txt").write_text("\n".join(p) + "\n")
local = [line(2, a, "out/ticket-local", "", 8) + f" >> out/ticket-local/t2-{a}.log 2>&1" for a in ARMS]
(out / "local_t2.sh").write_text("#!/bin/bash\n# t2 (ceiling task) on the laptop; resumable\ncd \"$(dirname \"$0\")/../../..\"\n"
                                 "export LIBERO_CONFIG_PATH=$PWD/out/config/libero MUJOCO_GL=egl HF_HUB_OFFLINE=1\n"
                                 "mkdir -p out/ticket-local\nP=/home/lab/miniforge3/envs/vla-libero/bin/python\n"
                                 + "\n".join(l.replace("python ", "$P ", 1) for l in local) + "\n")
print(sorted(p.name for p in out.iterdir()))
