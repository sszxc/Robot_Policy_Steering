# Resource audit: evaluation array 64364739

Queried ASU Sol with `sacct` on 2026-10-02. These are actual allocations,
not only the defaults in the submission scripts.

Each task: htc/public, one node, 4 CPUs, 32768 MiB host memory,
one full A100, 120-minute limit. Array: ten tasks, at most two concurrent.
Task 0's startup log identifies NVIDIA A100-SXM4-80GB, with MIG disabled.

| Task | Elapsed | Total CPU time | Batch MaxRSS (MiB) |
| --- | --- | --- | --- |
| 0 | 00:45:25 | 00:38:53.811 | 7981.83 |
| 1 | 00:41:53 | 00:35:12.718 | 5989.12 |
| 2 | 00:39:12 | 00:33:41.842 | 7881.73 |
| 3 | 00:26:57 | 00:24:48.287 | 6420.18 |
| 4 | 00:43:38 | 00:38:34.128 | 6244.52 |
| 5 | 00:28:00 | 00:25:03.355 | 6552.44 |
| 6 | 00:37:11 | 00:31:10.785 | 8392.14 |
| 7 | 00:36:06 | 00:34:17.944 | 5579.65 |
| 8 | 00:44:16 | 00:39:20.859 | 4145.46 |
| 9 | 00:33:28 | 00:31:23.757 | 5160.01 |

Totals: 6.2683 allocated GPU-hours; 25.0733 allocated CPU core-hours;
5.5410 consumed CPU core-hours. CPU efficiency is 22.10%, averaging
0.884 busy cores per task. This supports trying 2 CPUs; it does not establish
that 1 CPU would preserve throughput or cover peak demand.

Recorded batch MaxRSS spans 4.05–8.20 GiB, or 12.7–25.6% of requested
memory. MaxRSS is Slurm's recorded task peak, not a complete measurement of
all process and cache memory. The repo's HPC quick guide records a 24000 MB
minimum host-memory request for GPU jobs; verify current policy before
reducing memory. A 24G request is a candidate for validation.

All tasks finished within 45m25s. A 60-minute limit is a candidate for the
same workload. Resources are released when jobs finish; the unused portion
of a time limit is not additional occupied GPU time.

GPU utilization and GPU memory usage were absent from the queried Slurm
TRES usage fields. The startup nvidia-smi snapshot precedes model loading;
its 0% utilization is not evidence of idle GPUs during evaluation. Collect
periodic GPU telemetry during a future run before judging GPU efficiency.

Other jobs queried:

- Smoke 64361608: 4 CPUs, 32 GiB, 1 A100; elapsed 11m37s; CPU time
  4m25.983s; batch MaxRSS 8489.48 MiB.
- Cancelled serial evaluation 64361646: same resources; 4-hour limit;
  elapsed 34m44s; CPU time 33m01.014s; batch MaxRSS 6184.67 MiB.
  This occupied about 0.579 GPU-hours and did not complete the evaluation.
- Merge 64364771: 1 CPU, 4 GiB, no GPU; 10-minute limit; elapsed 9s;
  CPU time 0.748s; batch MaxRSS 35.13 MiB. Memory is oversized, but the
  absolute resource cost is small.

GPU jobs report `billing=40` in AllocTRES. Reducing CPU or host memory may
not reduce billing allocation when the GPU dominates; verify the partition's
billing weights before claiming accounting savings.

Query used:

```bash
sacct -j 64364739,64364771,64361608,64361646 --units=M --parsable2 \
  --format=JobID,State,ElapsedRaw,TimelimitRaw,AllocCPUS,ReqMem,AllocTRES,TotalCPU,MaxRSS,TRESUsageInAve
```
