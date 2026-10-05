# SmolVLA / LIBERO-10 baseline

Policy `HuggingFaceVLA/smolvla_libero`, relative actions, seed 1000, 10 tasks x 10
episodes. Pinned revisions: `experiments/libero/baselines/local-baseline.json`.

| Where | Success | Notes |
| --- | --- | --- |
| Laptop (`/home/lab/miniforge3/envs/vla-libero`) | 39/100 | ~9668 s serial; raw: `out/libero/smolvla/vanilla/libero10-full-20260930/` |
| Sol A100 array `64364739` | 31/100 | per-task: [full-array-64364739.json](smolvla_libero/full-array-64364739.json) |

One run each, so the 8-point gap alone doesn't indicate a mismatch.

Env: Python 3.12.14, LeRobot 0.6.2 (this subtree), torch 2.11.0+cu130,
hf-libero 0.1.4, MuJoCo 3.8.1, robosuite 1.4.0, transformers 5.5.4, torchcodec 0.11.1.
Sol env `/scratch/xzhan742/thesis/envs/smolvla-libero-py312-v3` built by `setup.sbatch`.

## Lessons from setup

- torchcodec 0.11.1 needs FFmpeg 4–8 (setup pins `ffmpeg=8.*`).
- The pure CUDA pip index lacks some pins: use PyPI + `download.pytorch.org/whl/cu130`.
- Offline eval needs config and tokenizer pointed at the local backbone snapshot (`prepare_policy.py`).
- Request `a100:1` explicitly for EGL rendering (avoid MIG slices).
- ~5 min/episode on one A100 at batch 1. Serial 100 episodes exceed htc's 4 h, hence the 10-task array.
- Per array task (4 CPU, 32 G, 1 A100): 27–45 min, MaxRSS 4–8 GiB, ~0.9 busy cores.
  2 CPU / 24 G / 1 h is a reasonable request to try. GPU utilization was not measured.
