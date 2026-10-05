# Project architecture

Two domains in one repo: policy control-space research (SmolVLA/LIBERO) and the
UR5 + HX5 robotics stack. Laptop and HPC share the Git checkout, not the Python env.

```text
thesis/
├── research/
│   ├── pretrained/prepare_policy.py   link pinned HF snapshots into an offline policy dir
│   └── evaluation/                    provenance, verify_eval, aggregate_eval
├── experiments/libero/
│   ├── run_libero10_smolvla.sh        generic lerobot-eval launcher (LIBERO-10)
│   ├── baselines/local-baseline.json  pinned Hub revisions (policy, backbone, assets)
│   └── object_parallel/               self-contained: 10 Object tasks on one A100
├── platforms/
│   ├── paths.py                       repo/asset/data/output paths (env-var overrides)
│   ├── hpc/                           runtime.sh, gpu_preflight.py, slurm/smolvla_libero/*.sbatch
│   └── environments/                  SmolVLA pip constraints, robotics versions
├── robotics/
│   ├── sim/                           MuJoCo scene composition, Sim runner, hand postures, viewer
│   ├── policies/scripted/             one script per hand task
│   ├── policies/rl/                   MJX envs, brax PPO train/eval
│   ├── teleop/                        Leap -> hand retargeting, record/replay
│   └── hardware/ur5/                  ur_rtde examples
├── assets/mujoco/                     hand/arm/object MJCF (UR5e model for now)
├── ros2_ws/src/thesis_scene/          UR5 Gazebo + MoveIt (see ros2_ws/CHEATSHEET.md)
├── lerobot/                           frozen LeRobot subtree (source + packaging only)
├── tests/
├── docs/                              usage, notes (hand task results), HPC, results
└── skills/ASU_Sol_HPC/                HPC skills, read on demand
```

Ignored locally: `out/` (outputs), `data/`, `ros2_ws/{build,install,log}`, `.local-backups/`.

## Conventions

- Keep LIBERO-specific config in `experiments/`; don't force LIBERO and UR5/HX5
  into a shared env API until a concrete integration needs it.
- `lerobot/` is a squashed subtree of upstream `e0d50211`, frozen (no upstream
  pulls; non-source files removed). Edit it in place, in separate commits. Relevant:
  SmolVLA, possibly Pi0.5 / GR00T later, and the LIBERO env.
- Install the root package with `pip install --no-build-isolation --no-deps -e .`
  into an existing env; LeRobot is installed separately in the SmolVLA env.
- `platforms/paths.py` env vars: `THESIS_REPO_ROOT`, `THESIS_DATA_ROOT` (`data`),
  `THESIS_OUTPUT_ROOT` (`out`), `THESIS_UR5_IP` (`192.168.1.132`). HPC jobs source
  `platforms/hpc/runtime.sh`, which sets scratch values and `THESIS_PYTHON_ENV`.
- Each LIBERO run gets a new output dir with `provenance.json` (commit, dirty,
  run args); lerobot's `eval_info.json` is the result.
- Git holds source, assets, configs, docs and small curated results; never
  checkpoints, videos, datasets or raw logs. Back up important scratch outputs.
