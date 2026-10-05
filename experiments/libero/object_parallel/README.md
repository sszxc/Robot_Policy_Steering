# LIBERO-Object: ten tasks sharing one A100

One allocation runs ten independent `run_task.sh` processes (Object tasks 0–9) on
the same GPU, 10 episodes each by default. Goal: measure whether overlapping tasks
improves single-GPU throughput vs. the one-task-per-GPU array. Separate processes
isolate SmolVLA action queues, at the cost of ten model copies in VRAM.

```bash
cd ~/Coding/thesis && mkdir -p /scratch/xzhan742/thesis/logs
sbatch experiments/libero/object_parallel/parallel10.sbatch          # 10 ep/task, batch 1
sbatch experiments/libero/object_parallel/parallel10.sbatch 1 1      # quick check
sbatch experiments/libero/object_parallel/parallel10.sbatch 10 2     # 2 envs per task
```

Args: episodes per task, batch size per task (must divide episodes). Batch size
changes policy sampling, so keep it fixed across comparisons. Request:
htc/public, 1 A100, 12 CPU, 64 G, 4 h.

Outputs in `/scratch/xzhan742/thesis/results/smolvla/vanilla/liberoobj-parallel-JOB_ID/`:
`task-N/` (eval + videos), `summary.json` (success, episodes/min; only if all
tasks pass), `task-status.tsv`, `logs/task-N.{out,err}`, and `resources/`
(`gpu.csv` sampled every 5 s, `processes.csv`, `gpu-start.txt`).

Review notes: `nvidia-smi` may list unallocated GPUs (match the UUID); process
CPU% is a lifetime average; samples include model loading. Also check
`seff JOB_ID` and `sacct -j JOB_ID`.
