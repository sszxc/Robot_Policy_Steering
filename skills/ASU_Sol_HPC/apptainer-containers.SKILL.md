---
name: apptainer-containers
description: "Build and run Apptainer/Singularity containers on an HPC cluster: convert a Docker image, write a definition file, bind scratch and project paths, pass GPUs through with --nv, and run a container correctly inside a Slurm job."
compatibility: opencode
metadata:
  audience: "researchers"
  runtime: "apptainer"
---

## What I do

- Turn a Docker image or a definition file into a runnable `.sif`
- Run containers inside Slurm jobs with the right bind mounts and GPU passthrough
- Debug "it works in Docker but not on the cluster"

## When to use me

Use me when software will not build natively, when a published pipeline ships as
a Docker image, or when an analysis must stay reproducible for a paper.

## Safety

Containers get **read-only binds by default** — a container with your data
mounted read-write can overwrite it just as easily as you can. I write new
images to new filenames rather than overwriting an existing `.sif`, and I never
bind a whole filesystem root or run anything with elevated privileges.

## Ground rules

- Clusters run **Apptainer** (formerly Singularity), not Docker. There is no
  daemon and no root — containers run as you, with your UID.
- Building usually needs `--fakeroot` and is often only allowed on a login node
  or a dedicated build host. Pulling a prebuilt image always works.
- The image is a single file. Keep `.sif` files on scratch or project space, not
  in `$HOME`.

## Pull an existing image

```bash
module load apptainer     # name varies; check 'module avail apptainer|singularity'
cd /scratch/$USER/images
apptainer pull pytorch.sif docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
```

## Build from a definition

```
Bootstrap: docker
From: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

%post
    apt-get update && apt-get install -y --no-install-recommends git
    pip install --no-cache-dir numpy scipy torch==2.4.0

%environment
    export PYTHONUNBUFFERED=1

%runscript
    exec python "$@"

%labels
    Maintainer  you@example.edu
    Version     1.0.0
```

```bash
apptainer build --fakeroot analysis.sif analysis.def
```

Pin versions in `%post`. An unpinned build is not reproducible, which defeats
the purpose of containerising it.

## Run it in a job

```bash
#!/bin/bash
#SBATCH --account=<allocation>
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load apptainer

apptainer exec --nv \
  --bind /data/project:/data/project:ro \
  --bind /scratch/$USER/proj/results:/results \
  /scratch/$USER/images/pytorch.sif \
  python /data/project/train.py --out /results
```

- `--nv` passes the NVIDIA driver through. Without it, CUDA is invisible inside.
- Only `$HOME`, the current directory, and `/tmp` are bound by default —
  everything else needs an explicit `--bind`.
- Add `:ro` to every bind except the one directory the job writes results to.
- `--cleanenv` gives a predictable environment; add it when host variables leak in.
- Use `--writable-tmpfs` if the software insists on writing inside the image — the
  writes land in memory and vanish at exit, leaving the image untouched.

## Common failures

| Symptom | Cause |
| --- | --- |
| `No such file or directory` for data that exists | Path not bound into the container |
| `CUDA driver version is insufficient` | Missing `--nv`, or image CUDA newer than the host driver |
| MPI ranks all report rank 0 | Container MPI does not match the host MPI; use the host's via bind |
| Build fails with a permission error | Build needs `--fakeroot`, or must happen on a build host |
| Image is huge | Use a `-runtime` base rather than `-devel`, and `--no-cache-dir` for pip |

Record the exact image digest (`apptainer inspect`) in the project README so the
run can be reproduced later.
