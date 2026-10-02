---
name: gpu-efficiency
description: "Measure and improve GPU utilization for training and inference jobs: find idle or under-fed GPUs, fix input pipeline bottlenecks, choose batch size and precision, and decide whether multi-GPU is justified before requesting it."
compatibility: opencode
metadata:
  audience: "researchers"
  hardware: "gpu"
---

## What I do

- Measure real GPU utilization for a running or finished job
- Find why a GPU is idle and fix the cause
- Decide whether more GPUs are justified

## When to use me

Use me before requesting additional GPUs, when a training run feels slow, or
when a job has been flagged as idle.

## Safety

Measurement only. I read utilization and job accounting; I never cancel a job or
release an allocation on your behalf, even one that looks idle — it might be
someone's checkpointing run or a deliberate warm-up.

## Measure first

```bash
# On the compute node (via an interactive session or srun --jobid=<id> --pty bash)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5

# After the job
seff <jobid>
```

Interpretation:

| Observation | Meaning |
| --- | --- |
| Utilization ~0% for minutes | The GPU is allocated and doing nothing — fix or release it |
| Utilization spiky 10–40% | Input pipeline starved: the GPU is waiting for data |
| Utilization 85–100% | Healthy; compute is the bottleneck |
| Memory used ≪ total | Batch size can grow, or a smaller GPU would do |

**Never conclude anything from a single instantaneous `nvidia-smi` reading.**
Sample over at least a few minutes of steady-state work.

## The usual causes of an idle GPU

1. **Data loading.** Too few workers, images decoded on the fly, or a dataset of
   millions of small files on a shared filesystem. Raise `num_workers`, enable
   `pin_memory` and `persistent_workers`, and pack the data into shards
   (WebDataset, TFRecord, HDF5, Parquet) on scratch.
2. **Small batches.** Grow the batch until memory is ~85% used, then scale the
   learning rate accordingly.
3. **Full precision.** Enable mixed precision (`torch.autocast` + `GradScaler`,
   or `bf16` where supported). It is usually the single largest win.
4. **Synchronous logging and validation.** Per-step `.item()`, printing, or a full
   validation pass every epoch stalls the device. Log every N steps.
5. **A held allocation nobody is using.** An interactive session left open
   overnight burns the allocation and blocks other researchers.

## Before requesting more GPUs

Ask, and answer with numbers:

- Is a single GPU above ~85% utilization for the whole run? If not, more GPUs
  multiply the idle time instead of shortening the job.
- Does the model actually fit? Single-GPU-plus-gradient-accumulation is simpler
  and often as fast as two GPUs.
- Have you measured scaling? Run 1 → 2 → 4 GPUs on a short job and compare
  samples/second. Below ~70% scaling efficiency, the extra GPUs are wasted.

## Site policy

Idle GPU allocations are tracked and flagged automatically. Treat a notice as a
signal to fix the pipeline or shrink the request — not as something to appeal.
Check utilization roughly 30 minutes into a long run so a wasted job is caught
early rather than at hour twelve.
