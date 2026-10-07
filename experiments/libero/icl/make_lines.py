"""ICL round lines files for platforms/hpc/slurm/steer_lines.sbatch (run with PAR=2): one A100 job
per task, VLM and CEM side by side (3 repeats each, B = 3 x K = 24).
python experiments/libero/icl/make_lines.py OUT_DIR
"""
import sys
from pathlib import Path

out = Path(sys.argv[1])
INITS = {0: 4, 7: 6, 9: 3, 4: 7}  # hard inits from steer2 (z = 0 succeeds 0-2 of 8 noise reps)
R = "/scratch/xzhan742/thesis/results/smolvla/icl"
run = "python -m research.optimize --policy $POLICY --repeats 3 --k 8 --budget 30 --batch-size 24"
out.mkdir(parents=True, exist_ok=True)
for t, i in INITS.items():
    lines = [f"{run} --task-id {t} --inits {i} --method {m} --output-dir {R}/t{t}/{m}" for m in ("vlm", "cem")]
    (out / f"icl-t{t}.txt").write_text("\n".join(lines) + "\n")
print(sorted(p.name for p in out.iterdir()))
# Follow-up: VLM with a per-chain signed permutation of z (removes the shared axis-order bias), 2 tasks per job
for name, ts in (("a", (0, 7)), ("b", (9, 4))):
    lines = [f"{run} --task-id {t} --inits {INITS[t]} --method vlm --perm --output-dir {R}/t{t}/vlm-perm" for t in ts]
    (out / f"icl-perm-{name}.txt").write_text("\n".join(lines) + "\n")
