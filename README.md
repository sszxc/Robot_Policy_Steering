# Robot Policy Steering

Find a compact control space over pretrained robot policies (SmolVLA on LIBERO
first). The UR5 + HX5 sim / RL / teleop stack lives in the separate HX5-Lab repo.

- [Architecture](docs/architecture.md)
- [Usage](docs/usage.md)
- [ASU Sol quick reference](docs/ASU_Sol_HPC_quick.md)
- [SmolVLA/LIBERO baseline](docs/results/smolvla_libero_baseline.md)

```bash
# In the SmolVLA env (editable lerobot[smolvla,libero]):
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest tests -q
TASK_IDS='[0]' EPISODES=1 bash experiments/libero/run_libero10_smolvla.sh
```
