# Architecture migration — 2026-10-02

Both machines started at `d0a334bce457868332f3ed1d4aa5a7ccc3031c35`.
Local working changes and untracked docs, skills and HPC source were captured in
baseline commit `f19a925b`. Robotics reorganization is commit `bf1bbf45`.

## Preservation

- Local `.local-backups/architecture-20261002/source-before.tar.gz` contains
  tracked source plus the original docs, skills, HPC tools and raw result backup.
  `changes.patch` and `status.txt` retain the initial Git state.
- HPC `.local-backups/architecture-20261002/untracked-before.tar.gz` preserves
  the original untracked `hpc/` tree and file `0`; its Git state is also recorded.
- HPC Python tools, Slurm scripts, constraints and baseline JSON matched the local
  versions. Its README described older queued jobs; the local README included
  newer completed outcomes, retained in `docs/results/smolvla_libero_baseline.md`.
- Raw local result logs, the environment archive and sample video now live in
  ignored `out/archive/smolvla_libero/`. Curated JSON summaries and the resource
  audit are tracked under `docs/results/smolvla_libero/`.
- No Git history rewrite, dependency upgrade, robot model replacement or hardware
  execution was performed. Future control, adaptation, HX5 and IL modules are
  documented locations rather than empty implementations.

## Local validation

- Before migration: 28 existing tests passed using `/home/lab/miniforge3/bin/python`.
- After migration: 32 tests passed, including hardware import isolation, path
  overrides, launcher provenance/overwrite protection and old-format aggregation.
- Editable installation completed with `--no-build-isolation --no-deps`.
- Pinch scripted task succeeded with 0.149 m rise from the repo and from `/tmp`.
- Rubik MJX smoke (`--envs 2 --steps 2`) completed on CPU with no NaNs. This
  environment reported no accessible CUDA device and no optional Warp backend;
  GPU training throughput was not validated by that smoke.
- All seven project shell entrypoints passed `bash -n`.

HPC validation is recorded below after synchronization and compute-node checks.
