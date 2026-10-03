# Environment records

Use separate environments for robot simulation/MJX, laptop hardware, and
SmolVLA/LIBERO. Installing the root package does not install those stacks.

- `smolvla_libero/`: the existing pinned constraints and full local package
  records. HPC uses `/scratch/xzhan742/thesis/envs/smolvla-libero-py312-v3`.
- `robotics_sim/local-observed.json`: versions observed in the prescribed local
  `/home/lab/miniforge3/bin/python` environment during migration.
- `robotics_local/local-observed.json`: relevant device distributions visible in
  that environment; missing distributions are null. Leap SDK/ROS system setup
  must also be recorded when a hardware environment is commissioned.

Observed records are not complete solver locks. Preserve working environments;
do not upgrade packages as part of directory migration. After installing the
appropriate stack, install the root checkout editable without dependencies.
