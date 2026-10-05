# Object evaluation: ten tasks sharing one A100

This evaluates the pinned pretrained checkpoint; it does not train or modify it.
One Slurm allocation launches ten independent policy processes, one per Object
task (0–9). All inherit the same allocated GPU. Each task runs ten episodes
sequentially by default: 100 episodes and 100 verified videos overall.
Independent processes isolate SmolVLA action queues and reset state. Ten model
copies consume more VRAM than a shared model with batched inference. This first
run measures whether overlapping ten tasks improves single-GPU throughput;
it does not assume that the models and renderers fit or that throughput scales
linearly. Failures are reported, with partial outputs retained.

## Submit on Sol

Sync the new files and the modified `research/evaluation/provenance.py` to
`~/Coding/thesis` first. The existing pinned environment and models are reused.
Ensure `/scratch/xzhan742/thesis/logs` exists before submitting.

```bash
cd ~/Coding/thesis
mkdir -p /scratch/xzhan742/thesis/logs
sbatch --test-only experiments/libero/object_parallel/parallel10.sbatch
sbatch experiments/libero/object_parallel/parallel10.sbatch
```

The initial request is one A100, 12 CPU cores, 64 GB host memory, and a four-hour
limit on `htc/public`, account `grp_hbenamor`. Account/partition settings follow
the project's verified October 1 records and were reconfirmed by read-only
cluster queries during preparation on October 4: `grp_hbenamor` has `public`
QoS, and `htc` is up with a four-hour limit and full `gpu:a100` resources.

Optional positional arguments are episodes per task and batch size per task:

```bash
# Short validation: still all ten tasks, but one episode per task.
sbatch experiments/libero/object_parallel/parallel10.sbatch 1 1
# Ten tasks, two environments per task: 20 concurrent environments.
# Increase CPU/memory requests only if measurements justify it.
sbatch experiments/libero/object_parallel/parallel10.sbatch 10 2
```

Batch size must divide episode count. The default is `10 1`; increasing the
batch changes policy random sampling and may change results even at fixed seed.
Keep batch size, seed, and reset settings fixed for subsequent policy comparisons.

## Outputs and resource review

Results live under
`/scratch/xzhan742/thesis/results/libero-object-parallel-JOB_ID/`:

- `task-0/` through `task-9/`: provenance, evaluation metrics, videos, verification.
- `summary.json`: combined success rate and episodes/minute, written only on success.
- `task-status.tsv`: each task PID and exit code.
- `logs/task-N.out`, `logs/task-N.err`: separate task logs.
- `resources/gpu.csv`: GPU utilization, VRAM, and power sampled every five seconds.
- `resources/processes.csv`: launcher descendants' CPU lifetime averages, RSS, threads.
- `resources/gpu-start.txt`: actual GPU model and available VRAM.

GPU rows include UUIDs because `nvidia-smi` can expose more physical GPUs than
the allocation. Match the allocated device when reviewing. Process CPU percentages
are `ps` lifetime averages, not instantaneous usage; RSS can double count shared
pages. Monitoring includes model loading and video verification. Inspect steady
rollout periods separately when assessing GPU utilization.

After completion, also collect Slurm accounting (replace `JOB_ID`):

```bash
seff JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,ReqMem,AllocTRES
```

Compare wall time and episodes/minute, alongside utilization and VRAM. The first
run has no measured speedup yet. Preserve important results off scratch.
