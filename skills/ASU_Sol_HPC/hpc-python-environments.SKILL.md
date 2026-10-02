---
name: hpc-python-environments
description: "Build reproducible Python environments on a shared HPC cluster using environment modules and mamba/conda: where to put an environment, how to keep home quota intact, how to pin dependencies, and how to make a batch job activate the environment reliably."
compatibility: opencode
metadata:
  audience: "researchers"
  language: "python"
---

## What I do

- Create and pin a Python environment that works on login and compute nodes
- Fix "works interactively, fails in batch" activation problems
- Keep environments off the home quota and out of the way of other users

## When to use me

Use me when setting up a project's dependencies on a cluster, or when an import
fails inside a job that works in an interactive shell.

## Safety

I only ever **create a new environment**. I do not remove, rename, or install
into an existing one that other work depends on, I do not clean package caches,
and I never use `sudo` or write into a shared or module-provided environment. If
an environment is broken, the fix is a fresh one beside it, not surgery on the
old one.

## Find what is already installed

```bash
module avail            # everything available
module spider pytorch   # search across versions
module load mamba/latest
```

Prefer a site-provided module over a hand-built install: it is compiled against
the cluster's MPI, CUDA, and interconnect. Only build your own when the module
does not exist or the version is wrong.

## Create the environment

```bash
module purge
module load mamba/latest

# Put environments and package caches on scratch/project space, NOT in $HOME —
# a conda environment is tens of thousands of small files and will blow a home quota.
export CONDA_PKGS_DIRS=/scratch/$USER/.conda/pkgs
mamba create -p /scratch/$USER/envs/myproj python=3.11 -y
source activate /scratch/$USER/envs/myproj

mamba install -c conda-forge numpy scipy pandas matplotlib -y
# Install the CUDA build that matches the cluster's driver:
mamba install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
```

Rules:

- Install compiled/scientific packages with `mamba` first, then `pip` for the rest.
  Mixing in the other order breaks solver-managed libraries.
- Never `pip install --user` on a cluster; it leaks into every environment.
- One environment per project. Shared "everything" environments rot.

## Make batch jobs use it

A batch shell is non-interactive and does not source your interactive profile,
so activation must be explicit inside the script:

```bash
module purge
module load mamba/latest
source activate /scratch/$USER/envs/myproj
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False` in the job but `True` interactively,
the job did not request a GPU (`--gres=gpu:1`) or loaded a CPU-only build.

## Pin it

```bash
mamba env export --from-history > environment.yml   # what you asked for
mamba list --explicit > environment.lock.txt        # exactly what you got
pip freeze > requirements.txt
```

Commit `environment.yml` for humans and the lock file for reproduction. Record
the module versions the environment was built against — they are part of the
environment even though conda does not capture them.

## Jupyter

```bash
source activate /scratch/$USER/envs/myproj
python -m ipykernel install --user --name myproj --display-name "myproj"
```

The kernel then appears in the portal's notebook sessions.

## Common failures

| Symptom | Cause |
| --- | --- |
| `Disk quota exceeded` while installing | Environment or package cache is in `$HOME` |
| `ModuleNotFoundError` only in batch | `source activate` missing from the script |
| CUDA available interactively, not in the job | No `--gres=gpu:1`, or a CPU-only build |
| Solver hangs for many minutes | Use `mamba`, and pin the Python version explicitly |
| Env works today, breaks after a cluster update | No lock file; regenerate and commit one |
