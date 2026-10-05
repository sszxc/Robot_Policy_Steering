# Robot Policy Steering

Find a compact control space over pretrained robot policies (SmolVLA on LIBERO
first), plus simulation, teleop and hardware tooling for a UR5 + HX5-D20-MRT-Right.

- [Architecture](docs/architecture.md)
- [Robot sim / RL / teleop / UR5 commands](docs/usage.md)
- [ASU Sol quick reference](docs/ASU_Sol_HPC_quick.md)
- [SmolVLA/LIBERO baseline](docs/results/smolvla_libero_baseline.md)

```bash
/home/lab/miniforge3/bin/python -m pip install --no-build-isolation --no-deps -e .
/home/lab/miniforge3/bin/python -m pytest tests -q
# In the SmolVLA env (editable lerobot[smolvla,libero]):
TASK_IDS='[0]' EPISODES=1 bash experiments/libero/run_libero10_smolvla.sh
```
