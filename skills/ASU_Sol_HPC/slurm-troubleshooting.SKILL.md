---
name: slurm-troubleshooting
description: "Diagnose Slurm jobs that will not start, died early, or produced nothing: read pending reason codes, interpret sacct/seff exit states, and identify OOM kills, walltime kills, quota failures, and account or QOS problems."
compatibility: opencode
metadata:
  audience: "researchers"
  scheduler: "slurm"
---

## What I do

- Work out why a job is stuck in `PENDING`
- Work out why a job failed, and whether it was the code, the resources, or the cluster
- Turn the finding into a concrete change to the submit script

## When to use me

Use me when a job is not running, ended unexpectedly, or produced no output.

## Safety

Every command here is read-only. I do not cancel, requeue, hold, or resubmit
jobs, and I do not delete logs or output — a failed job's artifacts are the
evidence. I report what I found and the command you can run to act on it.

## Step 1 — Get the facts

```bash
squeue -u "$USER" -o "%.10i %.9P %.20j %.8T %.10M %.6D %R"   # state + reason
scontrol show job <jobid>                                     # full request
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,ReqMem,MaxRSS,AllocTRES%40
seff <jobid>                                                  # efficiency summary
```

Never diagnose without at least the state, the reason code, and the resource
request in front of you.

## Step 2 — Pending reason codes

| Reason | What it means | What to change |
| --- | --- | --- |
| `Priority` | Healthy queueing; other jobs are ahead | Nothing is broken. A shorter walltime improves backfill odds |
| `Resources` | The requested shape is scarce right now | Fewer GPUs/nodes, less memory, shorter time |
| `QOSMaxJobsPerUserLimit` | At a per-user QOS limit | Wait for a running job to finish, or throttle an array with `%N` |
| `AssocGrpBillingMinutes`, `AssocGrp*` | The allocation is exhausted or capped | Talk to the allocation owner; do not just resubmit |
| `ReqNodeNotAvail` | Usually a maintenance reservation ahead | The job cannot finish before the outage — shorten the walltime |
| `PartitionConfig`, `InvalidQOS` | The partition/QOS pair is not valid for this account | Re-check the association listing |
| `Dependency` | Waiting on another job | Check the dependency actually completed successfully |

A job that has been pending for hours with `Priority` on a busy cluster is
normal. Say so instead of inventing a fix.

## Step 3 — Failure signatures

| Symptom | Cause | Fix |
| --- | --- | --- |
| `State=TIMEOUT`, output stops mid-run | Hit `--time` | Raise walltime and add checkpointing |
| `State=OUT_OF_MEMORY`, or `MaxRSS` ≈ `ReqMem` | OOM kill | Raise `--mem`, or stream/shard the data |
| `CUDA out of memory` in the log | GPU memory, not host memory | Smaller batch, gradient accumulation, mixed precision |
| `ExitCode 127` | Command not found | `module load` missing, or the env was never activated |
| `ExitCode 1` immediately | The program itself failed | Read the first error in the log, not the last |
| `Disk quota exceeded` | Home or project quota is full | Write to scratch; clean checkpoints |
| Empty output file | Job never started, or the script `cd`'d somewhere unexpected | Check `sacct` state and the working directory |
| Works interactively, fails in batch | Batch shells do not read the interactive profile | Load modules and activate environments inside the script |

## Step 4 — Reproduce small

Before changing the batch script, reproduce the failure in a short interactive
allocation with the same modules and the same data path. A five-minute
interactive repro beats three hours of resubmitting.

## How I report back

1. **What happened** — the state, exit code, and the evidence line that proves it
2. **Why** — one sentence
3. **The fix** — the exact directive or code change, as a diff
4. **How to confirm** — the command that shows the fix worked

If the evidence does not identify a cause, I say that and list what to collect
next rather than guessing.
