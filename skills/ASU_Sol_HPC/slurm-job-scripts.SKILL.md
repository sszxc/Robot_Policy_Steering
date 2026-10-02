---
name: slurm-job-scripts
description: "Write and review Slurm sbatch scripts for the Sol and Phoenix clusters: account, partition, QOS, walltime, GPU and array directives, plus the resource right-sizing rules that keep a job out of the queue's slow lane."
compatibility: opencode
metadata:
  audience: "researchers"
  scheduler: "slurm"
---

## What I do

- Write `sbatch` scripts with a complete, correct directive header
- Convert an ad-hoc command into a batch job, an array job, or an interactive session
- Review an existing script for missing accounts, wrong partitions, and oversized requests

## When to use me

Use me whenever a job needs to be submitted, rewritten, or reviewed on an HPC cluster.

## Safety

I write and read files inside your project, and I run read-only cluster queries
(`sinfo`, `squeue`, `sacctmgr show`). I do not submit or cancel jobs, delete old
outputs, or touch anything outside the project directory — I hand you the
`sbatch` command and you run it.

## Always establish these first

Never guess these four. Ask the user, or read them off the cluster:

| Value | How to confirm |
| --- | --- |
| Account (allocation) | `sacctmgr show associations user=$USER format=account,qos,partition` |
| Partition | `sinfo -s` |
| QOS | the association listing above |
| Walltime limit | `sacctmgr show qos format=name,maxwall,maxtresperuser` |

Site defaults as a starting point, not as truth:

- **Sol / Phoenix**: partition `general`, QOS `public` (also `private`, `debug`, `class`).
- The `general` partition does not exist everywhere. On some clusters it is defined but has no nodes, so a job sent there is queued forever and never starts. Confirm with `sinfo -s` before submitting.

## Batch script skeleton

```bash
#!/bin/bash
#SBATCH --job-name=train-resnet
#SBATCH --account=<allocation>
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL

module purge
module load mamba/latest
source activate myenv

srun python train.py --epochs 10
```

## Array jobs

Prefer many small tasks over one long sequential job — the scheduler backfills
small tasks into gaps, so an array usually starts sooner and finishes earlier.

```bash
#SBATCH --array=1-200%20          # 200 tasks, at most 20 running at once
#SBATCH --output=%x-%A_%a.out     # %A = array job id, %a = task index

input=$(sed -n "${SLURM_ARRAY_TASK_ID}p" inputs.txt)
python process.py --input "$input"
```

Do not request a GPU in an array unless *every* task uses one.

## Interactive work

```bash
salloc --account=<allocation> --partition=general --qos=public \
       --time=2:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1
```

Interactive sessions are for debugging and profiling, not for training runs.
Exit as soon as the work is done — the allocation is charged until it is
released, and an idle interactive GPU is the fastest way to trip the efficiency
policy.

## Rules I hold to

1. **Request what the job uses.** Ask for measured numbers (`seff <jobid>` on a
   previous run) rather than round numbers that "look safe". Oversized requests
   wait longer and waste the allocation.
2. **Walltime is a limit, not an estimate.** Set it about 25% above the observed
   runtime. A too-short walltime kills the job; a wildly long one blocks backfill.
3. **One GPU until multi-GPU is proven.** Add GPUs only after showing that a
   single GPU is saturated.
4. **Checkpoint anything longer than the QOS walltime limit** and make the script
   resumable rather than asking for an exception.
5. **`module purge` first** so the job does not inherit whatever the login shell
   happened to have loaded.
6. **Write output to scratch, not to home.** Home quotas are small and a full home
   directory fails jobs in confusing ways.

## Before I hand a script back

- Every directive has a real value — no `<placeholder>` left in the file
- Paths are absolute or relative to a `cd "$SLURM_SUBMIT_DIR"`
- The script is committed alongside the code it runs
- I state the submit command explicitly: `sbatch script.sh`
