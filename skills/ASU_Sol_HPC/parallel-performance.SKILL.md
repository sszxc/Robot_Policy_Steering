---
name: parallel-performance
description: "Profile and optimise parallel scientific code (MPI, OpenMP, CUDA): measure before changing anything, run strong and weak scaling studies, interpret roofline and profiler output, and apply the optimisations that matter in the order that matters."
compatibility: opencode
metadata:
  audience: "researchers"
  topic: "performance"
---

## What I do

- Profile a serial or parallel code and identify the actual bottleneck
- Run and interpret strong/weak scaling studies
- Apply MPI, OpenMP, and CUDA optimisations in a defensible order

## When to use me

Use me when code is "too slow", when scaling is disappointing, or when a paper
needs a performance section.

## Safety

Benchmarks run **in a job, never on a login node** — a parallel run on a shared
login node degrades the cluster for everyone. Baseline timings and reference
outputs are kept, not overwritten: without the baseline there is no result.

## Non-negotiable order

1. **Make it correct.** Keep a reference result and a tolerance. An optimisation
   that changes the answer is a bug.
2. **Measure.** Profile before touching anything. Intuition about hotspots is
   wrong more often than it is right.
3. **Fix the biggest hotspot.** Amdahl's law: speeding up 5% of the runtime by
   10× buys you 4.7%.
4. **Re-measure.** Report the speedup against the baseline you recorded.

## Tools

```bash
# CPU
perf record -g ./app && perf report
gprof / valgrind --tool=callgrind
python -m cProfile -o out.prof script.py && python -m pstats out.prof

# MPI (inside a job allocation, never on a login node)
srun -n 64 ./app          # with a site MPI profiler: mpiP, Score-P, TAU, HPCToolkit
export MPICH_ENV_DISPLAY=1  # or OMPI equivalents, to confirm what actually ran

# GPU
nsys profile -o report ./app          # timeline: is the GPU even busy?
ncu --set full -o kernel ./app        # per-kernel counters, after nsys narrows it down
```

Profile a **representative but short** problem. A profile of a five-second toy
run tells you about startup, not about science.

## Scaling studies

- **Strong scaling**: fixed total problem, increasing ranks. Report speedup
  `T(1)/T(N)` and efficiency `T(1)/(N·T(N))`. Efficiency below ~70% marks the
  useful limit — that is the number to request in production jobs.
- **Weak scaling**: fixed work *per rank*, increasing ranks. Report efficiency
  `T(1)/T(N)`; flat is ideal.

Method notes that make the numbers credible: fixed node type, exclusive nodes,
at least three repeats reporting median and spread, timers around the compute
region excluding I/O and startup, and the compiler/MPI versions stated.

## Where the time usually goes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Scales to 8 ranks, then flattens | Communication dominates | Larger blocks, non-blocking + overlap, fewer collectives |
| One rank much slower | Load imbalance | Dynamic scheduling, better decomposition |
| Poor OpenMP speedup | False sharing or a serial region | Pad structures, widen the parallel region, check `OMP_PROC_BIND=close` |
| GPU kernel much slower than expected | Memory-bound, not compute-bound | Coalesce accesses, use shared memory, fuse kernels |
| Long stalls between kernels | Host–device transfers | Pinned memory, async copies, keep data resident |
| Everything is slow at start | I/O | Read once on rank 0 and broadcast, or use parallel I/O |

Use the roofline model to decide the ceiling: measure arithmetic intensity
(FLOP/byte) and check whether the kernel is memory-bound or compute-bound. Do
not micro-optimise arithmetic in a memory-bound kernel.

## Reporting

Always give: hardware, compiler and flags, MPI/CUDA versions, problem size,
repeat count, and the baseline. A speedup number with no baseline is not a
result.
