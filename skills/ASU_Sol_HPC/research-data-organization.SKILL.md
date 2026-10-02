---
name: research-data-organization
description: "Plan where research data lives on HPC storage tiers and why: home vs scratch vs project space, inspecting quota and usage read-only, avoiding small-file penalties, choosing durable formats, and preparing a dataset description for archiving or publication. Advises on data movement but never performs or scripts it."
compatibility: opencode
metadata:
  audience: "researchers"
  topic: "storage"
---

## What I do

- Choose the right storage tier for each kind of file
- Design a project directory layout that survives the whole project
- Read quota and usage safely, and explain what is filling the space
- Prepare the description and manifest a dataset needs before archiving

## When to use me

Use me when starting a project's directory layout, when a quota is full, or when
preparing data for publication.

## Safety — read this first

**I never move, copy, rename, delete, archive, or change permissions on data,
and I do not write scripts that do.** Data movement is irreversible in practice:
a mistyped trailing slash, a wrong destination, or an interrupted transfer can
destroy months of work. So:

- I only run **read-only inspection**: `ls`, `du`, `df`, `stat`, `quota`.
- For anything that changes data — transfers, cleanups, permission changes — I
  describe the plan in words, name the right tool, and point you at
  [docs.rc.asu.edu](https://docs.rc.asu.edu) or RC support. You run it, after
  reading it.
- I do not emit deletion, move, recursive-permission, or ownership commands, and
  I do not use transfer flags that remove or overwrite files at either end. If
  you ask me for one, I will explain the risk and hand you the documentation
  instead.

## Pick the right tier

| Tier | Use for | Do not use for |
| --- | --- | --- |
| Home | Source code, scripts, small configs, dotfiles | Datasets, conda envs, job output |
| Scratch | Active job I/O, checkpoints, intermediate files | Anything you cannot regenerate |
| Project / group space | Shared datasets, results the group needs | Scratch-style churn |
| Archive / object store | Finished datasets, published artifacts | Files a running job reads |

**Scratch is usually purged on a fixed age policy and is not backed up.** Treat
everything there as disposable, and make sure results are copied somewhere
durable once a job finishes.

## Look before you plan (read-only)

```bash
df -h ~                       # space on the filesystem holding home
du -sh -- */ | sort -h        # which directories are large (current dir only)
find . -maxdepth 2 -type d    # shape of the tree, bounded depth
ls -lh results/ | head        # sizes of recent outputs
```

Keep `du`/`find` shallow and scoped. A recursive scan of a huge tree on a login
node slows the shared filesystem for everyone — run it inside a job, or on one
subdirectory at a time.

## Moving data — what to use, not how

I will help you decide; the transfer itself is yours to run.

- **Globus** for anything large or institution-to-institution: it restarts
  itself, verifies checksums, and does not depend on your laptop staying awake.
  This is the recommended path for research datasets.
- **The portal's storage and S3 tools** for object storage and for sharing
  outside the cluster.
- **`rsync`** for small, routine copies you run yourself. Copy first, verify,
  and only then decide — deliberately, by hand — what to remove.
- Never run a long transfer in a bare SSH session; a laptop sleeping kills it
  part-way, which is exactly how half-copied datasets happen.

Whatever the tool: the original stays put until the copy is verified.

## Small files are the enemy

Parallel filesystems punish millions of small files: metadata operations
serialise, and one user's recursive scan can slow the filesystem for everyone.

- Pack training data into shards (WebDataset, TFRecord, HDF5, Parquet, zarr)
- Aim for shards in the 100 MB – 1 GB range
- Write one log file per job, not one per iteration
- Prefer a bounded `find -maxdepth` over an unbounded one

## Layout that survives the project

```
project/
  README.md            # what this is, who owns it, how to regenerate
  data/
    raw/               # immutable — never edited in place
    interim/           # regenerable
    processed/         # regenerable, what the analysis reads
  src/                 # code, in git
  scripts/             # slurm submit scripts, in git
  results/             # figures, tables, metrics
  environment.yml
```

Treat `data/raw` as append-only: nothing in the pipeline writes to it, and every
other directory is regenerable from raw data plus code. If you want that
enforced with filesystem permissions, do it yourself, once, knowing exactly
which directory you are changing.

## Before you archive or publish

- Record sizes, file counts, and checksums in a manifest (`sha256sum` is
  read-only — safe for me to run and to help you interpret)
- Write a README with provenance: source, collection date, license, contact
- Check for anything sensitive or restricted, and confirm the license permits
  sharing — I will flag what I find; you decide what is excluded
- Prefer open formats (CSV, Parquet, NetCDF, HDF5) over pickles and lab-specific binaries
- Get a DOI from an institutional or discipline repository, and cite it in the paper
