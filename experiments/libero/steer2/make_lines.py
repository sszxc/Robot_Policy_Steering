"""Steering round 2 lines files for platforms/hpc/slurm/steer_lines.sbatch (run with PAR=2).
stage1-t{T}-{a,b}: PCA fit + grids; stage2-t{T}: select, line scans, held-out, open-loop.
python experiments/libero/steer2/make_lines.py B OUT_DIR
"""
import sys
from pathlib import Path

B = int(sys.argv[1])
out = Path(sys.argv[2])
TASKS = (0, 7, 9, 4, 2)
R = "/scratch/xzhan742/thesis/results/smolvla/steer2"
RAND = "--w random --alpha 0.3"
GRID = f"--n-z 8 --scales=-2,-1,-0.5,0.5,1,2 --inits 8 --batch-size {B} --videos 1"
run = "python -m research.run --policy $POLICY"
out.mkdir(parents=True, exist_ok=True)
for t in TASKS:
    d = f"{R}/t{t}"
    pca = f"--w {d}/pca_fit/pca.pt --alpha 3.0"
    fit = (f"python -m research.fit_pca --policy $POLICY --task-id {t} --episodes 10 --init-start 40 --reps 3 "
           f"--batch-size 10 --output-dir {d}/pca_fit")
    a = [f"{fit} && {run} --task-id {t} {pca} {GRID} --zero-reps 0 --output-dir {d}/grid-pca",
         f"{run} --task-id {t} {RAND} --w-seed 0 {GRID} --zero-reps 8 --output-dir {d}/grid-random-s0"]
    b = [f"{run} --task-id {t} {RAND} --w-seed {s} {GRID} --zero-reps 0 --output-dir {d}/grid-random-s{s}" for s in (1, 2)]
    (out / f"stage1-t{t}-a.txt").write_text("\n".join(a) + "\n")
    (out / f"stage1-t{t}-b.txt").write_text("\n".join(b) + "\n")

    ws = {"random-s0": f"{RAND} --w-seed 0", "random-s1": f"{RAND} --w-seed 1",
          "random-s2": f"{RAND} --w-seed 2", "pca": pca}
    sel = " && ".join(f"python -m research.select_z {d}/grid-{w} --top 1 --out {d}/sel-{w}" for w in ws)
    line = "--scales=-3,-1.5,-0.75,-0.25,0.25,0.75,1.5,3 --inits 8 --zero-reps 1 --videos 0"
    val = "--scales 1 --inits 30 --init-start 10 --z-reps 2 --videos 1"
    lines = [f"{sel} && {run} --task-id {t} {ws['random-s0']} --z-file {d}/sel-random-s0-line.npy {line} "
             f"--batch-size {B} --output-dir {d}/line-random-s0",
             f"while [ ! -f {d}/sel-pca.json ]; do sleep 10; done; {run} --task-id {t} {pca} --z-file "
             f"{d}/sel-pca-line.npy {line} --batch-size {B} --output-dir {d}/line-pca"]
    for w, arg in ws.items():
        z0 = 2 if w == "random-s0" else 0
        lines.append(f"{run} --task-id {t} {arg} --z-file {d}/sel-{w}-val.npy {val} --zero-reps {z0} "
                     f"--batch-size {B} --output-dir {d}/val-{w}")
    oloop = " && ".join(f"python -m research.openloop --policy $POLICY --task-id {t} --inputs {d}/pca_fit/inputs.pt "
                        f"{arg} --n-z 8 --batch-size 32 --output-dir {d}/openloop-{w}" for w, arg in ws.items())
    lines.append(oloop)
    (out / f"stage2-t{t}.txt").write_text("\n".join(lines) + "\n")
print(sorted(p.name for p in out.iterdir()))
