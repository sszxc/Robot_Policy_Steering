# Project architecture

Policy control-space research (SmolVLA / Pi0.5 on LIBERO). Laptop and HPC share the Git
checkout, not the Python env. The UR5 + HX5 sim / RL / teleop stack lives in the separate
repo `~/Documents/HX5-Lab`; real-robot runs go through `~/Documents/ros2_ur`.

```text
Robot_Policy_Steering/
├── research/
│   ├── pretrained/prepare_policy.py   link pinned HF snapshots into an offline policy dir
│   ├── evaluation/                    provenance, verify_eval, aggregate_eval
│   ├── rollout.py                     load SmolVLA + LIBERO vec env once; batched episodes per init state
│   ├── inject/                        z injection configs: attach(policy, **args) -> (dim, set_z)
│   ├── fit_pca.py                     PCA of expert activations (for expert_film --w)
│   └── run.py                         steering entry: z=0 vs random z, videos; --check
├── experiments/libero/
│   ├── run_libero10_smolvla.sh        generic lerobot-eval launcher (LIBERO-10)
│   ├── baselines/local-baseline.json  pinned Hub revisions (policy, backbone, assets)
│   └── object_parallel/               self-contained: 10 Object tasks on one A100
├── platforms/
│   ├── paths.py                       repo/data/output paths (env-var overrides)
│   ├── hpc/                           runtime.sh, gpu_preflight.py, slurm/{steer,smolvla_libero/*}.sbatch
│   └── environments/                  SmolVLA pip constraints
├── lerobot/                           frozen LeRobot subtree (source + packaging only)
├── tests/
├── docs/                              usage, HPC, results
└── skills/ASU_Sol_HPC/                HPC skills, read on demand
```

Ignored locally: `out/` (outputs), `data/`, `.local-backups/`.

## Conventions

- Keep LIBERO-specific config in `experiments/`.
- `lerobot/` is a squashed subtree of upstream `e0d50211`, frozen (no upstream
  pulls; non-source files removed). Edit it in place, in separate commits. Relevant:
  SmolVLA, possibly Pi0.5 / GR00T later, and the LIBERO env.
- Install the root package with `pip install --no-build-isolation --no-deps -e .`
  into an existing env; LeRobot is installed separately in the SmolVLA env.
- `platforms/paths.py` env vars: `RPS_REPO_ROOT`, `RPS_DATA_ROOT` (`data`),
  `RPS_OUTPUT_ROOT` (`out`). HPC jobs source
  `platforms/hpc/runtime.sh`, which sets scratch values and `RPS_PYTHON_ENV`;
  GPU LIBERO jobs then call its `libero_runtime` (env activation, offline caches, EGL).
- Each LIBERO run gets a new output dir with `provenance.json` (commit, dirty,
  run args); lerobot's `eval_info.json` is the result.
- Git holds source, assets, configs, docs and small curated results; never
  checkpoints, videos, datasets or raw logs. Back up important scratch outputs.
