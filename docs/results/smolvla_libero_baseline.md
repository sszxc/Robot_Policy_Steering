# SmolVLA / LIBERO evaluation on ASU Sol

This document records the migration of the local SmolVLA/LIBERO evaluation
environment to ASU Sol. The user authorized installation, GPU validation and
full evaluation without separate approvals. Code remains in
`/home/xzhan742/Coding/thesis`, and environments, caches, models, results and
logs live under `/scratch/xzhan742/thesis`.

## Baseline and pinned inputs

- Local runtime: `/home/lab/miniforge3/envs/smolvla/bin/python`, Python 3.12.14.
- Editable LeRobot 0.6.2 from this checkout; local `pip check` passes.
- PyTorch 2.11.0+cu130, torchvision 0.26.0, hf-libero 0.1.4, MuJoCo 3.8.1,
  robosuite 1.4.0, transformers 5.5.4, torchcodec 0.11.1.
- Local and remote Git HEAD at setup: `d0a334bce457868332f3ed1d4aa5a7ccc3031c35`.
- Local reference: 100 episodes, 39 successes (39%), 9667.86 seconds.
- Local installed package versions and exact Hub model revisions are recorded
  in [`local-baseline.json`](../../experiments/libero/baselines/local-baseline.json). [`local-constraints.txt`](../../platforms/environments/smolvla_libero/local-constraints.txt) limits the HPC Python
  resolver to these versions for the `[smolvla,libero]` dependency graph.
- Sol has x86_64 Rocky Linux 8.10 with glibc 2.28; account `grp_hbenamor`,
  QoS `public`, `htc` full A100. System module: `mamba/latest`.

## Jobs and reproducible commands

| Job | Purpose | Outcome or current role |
| --- | --- | --- |
| `64306771` | First setup attempt | Failed: CUDA-only pip index lacks fixed `filelock` |
| `64322980` | PyPI + CUDA index install, FFmpeg 9 | Completed; snapshots cached |
| `64324198` | Fresh installation with FFmpeg 8.1.2 | Completed; `pip check` and torchcodec import passed |
| `64324587` | Fixed LIBERO asset revision to scratch | Completed; scratch asset symlink verified |
| `64324618` | GPU smoke preflight | EGL frame rendered; GL renderer string query caused false failure |
| `64325716` | Offline evaluation smoke | Model loaded; tokenizer still referenced Hub name |
| `64361608` | One-episode evaluation smoke | Completed; one video decoded, zero successes in this one episode |
| `64361646` | Single 100-episode job | Stopped after 34 minutes: 8 videos, too slow for 4-hour limit |
| `64364739` | Ten-task, 100-episode array | All ten tasks completed, exit code 0 |
| `64364771` | Aggregate metrics and verify 100 videos | Completed, exit code 0 |

The third setup (`setup.sbatch`) creates a fresh environment at
`/scratch/xzhan742/thesis/envs/smolvla-libero-py312-v3`. FFmpeg 8 is required
because torchcodec 0.11.1 ships libraries for FFmpeg major versions 4 through
8. Earlier setup environments are retained for diagnosis and are not used by
evaluation. All Python package pins remain unchanged.

The GPU job (`platforms/hpc/slurm/smolvla_libero/eval.sbatch`) runs
`bash experiments/libero/run_libero10_smolvla.sh` from the repository root, with the pinned model snapshots resolved locally and all
runtime data sent to scratch. `smoke` uses `TASK_IDS='[0]' EPISODES=1`. The
`array.sbatch` job calls task mode with 10 episodes for each task ID 0–9, at most
two concurrent A100 tasks. The smoke job's single episode required about 304
seconds of evaluation; the first serial full attempt was therefore stopped
before reaching the 4-hour HTC limit. The array preserves the per-task seeds
and episode counts of the original evaluation script. `aggregate_eval.py`
combines the verified per-task results and checks all 100 videos. GPU execution,
headless EGL rendering, actual LIBERO rollouts and video decoding have passed.

Useful remote read-only checks:

```bash
sacct -j 64364739,64364771 --format=JobID,State,ExitCode,Elapsed,MaxRSS
myjobs
tail -n 50 /scratch/xzhan742/thesis/logs/libero10-array-64364739_0.out
tail -n 50 /scratch/xzhan742/thesis/logs/libero10-array-64364739_0.err
```

Each successful task writes
`/scratch/xzhan742/thesis/results/libero10-taskN-64364739_N/verification.json`.
The aggregate job writes
`/scratch/xzhan742/thesis/results/libero10-full-array-64364739/summary.json`.
Scratch has no snapshots and inactive files are purged, so retain important
results elsewhere after evaluation.

## Verified result (2026-10-02)

The A100 task array completed 100 episodes: **31 successes (31%)**. The local
reference run had 39 successes (39%). This is one run on each machine, so the
8-point difference does not by itself establish a software or policy mismatch.
All 10 task jobs and the merge job exited 0. Each task wrote metrics for 10
episodes and 10 decodable MP4 videos. The local backup contains the
[summary](smolvla_libero/full-array-64364739.json), Slurm states (`out/archive/smolvla_libero/slurm-64364739.txt`),
environment manifests (`out/archive/smolvla_libero/environment-64324198.tar.gz`),
[smoke metrics](smolvla_libero/smoke-64361608.json), and a
sample smoke video (`out/archive/smolvla_libero/smoke-64361608.mp4`). All raw videos remain on
scratch under the task output paths listed in the summary. The data transfer
node connection timed out from this machine; these small artifacts were copied
through the login connection.

## Architecture migration

Source inputs were recorded in commit `f19a925b`. Both machines started at
`d0a334bc`; HPC scripts and Python tools matched byte for byte. The remote README
contained earlier queue statuses, while the local document contained completed
results; this document retains the newer verified results. The original remote
README is preserved in `.local-backups/architecture-20261002/untracked-before.tar.gz`
on HPC. Raw local artifacts moved to ignored `out/archive/smolvla_libero/`.
