# Robot Policy Steering

Discover a compact, expressive control space over pretrained robot policies,
starting with SmolVLA and LIBERO. The robotics branch develops simulation,
teleoperation and policy workflows for a **UR5 + HX5-D20-MRT-Right** system.
The current MuJoCo arm model remains UR5e pending a separate model validation.

## Start here

- [Architecture and directory tree](docs/architecture.md)
- [Robot simulation, RL and teleoperation commands](docs/usage.md)
- [ASU Sol quick reference](docs/ASU_Sol_HPC_quick.md)
- [SmolVLA/LIBERO baseline](docs/results/smolvla_libero_baseline.md)
- [LeRobot subtree maintenance](docs/lerobot_maintenance.md)

Use an existing environment; this reorganization does not upgrade dependencies:

```bash
/home/lab/miniforge3/bin/python -m pip install --no-build-isolation --no-deps -e .
/home/lab/miniforge3/bin/python -m pytest tests -q
/home/lab/miniforge3/bin/python -m robotics.policies.scripted.grasps --grasp pinch
```

SmolVLA/LIBERO uses its own environment and editable LeRobot installation:

```bash
python -m pip install --no-build-isolation --no-deps -e .
# Install lerobot[smolvla,libero] using the recorded constraints when building an environment.
TASK_IDS='[0]' EPISODES=1 bash experiments/libero/run_libero10_smolvla.sh
```

Each LIBERO run receives a new output directory and `provenance.json`. Existing
output directories are rejected to preserve earlier results.

Code and experiment configurations travel through Git. Datasets, environments,
weights and ordinary run outputs stay outside Git. Hardware runs and collection
stay on the laptop; training and evaluation can also use HPC.
