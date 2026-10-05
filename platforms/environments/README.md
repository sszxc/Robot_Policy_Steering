# Environment records

Separate environments; installing the root package doesn't install any stack.

- SmolVLA/LIBERO: `smolvla_libero/local-constraints.txt` (pip pins from the laptop
  env, used by `platforms/hpc/slurm/smolvla_libero/setup.sbatch`).
  HPC env: `/scratch/xzhan742/thesis/envs/smolvla-libero-py312-v3`.
- Robotics sim + devices: `/home/lab/miniforge3/bin/python` (3.13): mujoco 3.7.0,
  mujoco-mjx 3.7.0, jax/jaxlib 0.11.2 (cuda12), brax 0.14.2, numpy 2.4.4,
  scipy 1.17.1, pytest 9.1.1, pynput 1.8.2, ur-rtde 1.6.5, leapc-cffi (`leap` 0.0.1).
