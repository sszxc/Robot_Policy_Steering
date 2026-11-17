# Usage

Run commands from the repository root in the SmolVLA/LIBERO environment (laptop:
`/home/lab/miniforge3/envs/vla-libero/bin/python`; Sol: see [the quick reference](ASU_Sol_HPC_quick.md)).
Default outputs go under `out/` or `THESIS_OUTPUT_ROOT`. Headless rendering needs `MUJOCO_GL=egl`.

## SmolVLA / LIBERO

Select the SmolVLA/LIBERO environment with editable LeRobot installed, then:

```bash
TASK_IDS='[0]' EPISODES=1 bash experiments/libero/run_libero10_smolvla.sh
```

The launcher retains LIBERO-10 relative actions and seed 1000 by default.
`POLICY`, `TASK_IDS`, `EPISODES`, `SEED`, `EPISODE_LENGTH`, and `OUTPUT_DIR` can
override the run inputs. Default results receive a new directory under `out/libero/smolvla/vanilla/`;
an explicitly selected directory must also be new. Each run records provenance.
For HPC submission use [the quick reference](ASU_Sol_HPC_quick.md).

## Tests

```
python -m pytest tests -q
```
