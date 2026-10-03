# Project architecture

One repository contains two domains: policy control-space research and robotics.
Experiments connect these domains; platform code supplies deployment and runtime
settings. Both machines use the same Git commit but separate Python environments.

## Directory tree

`[future]` describes planned locations, not empty implementations.
`[ignored]` describes local/runtime content excluded from Git.

```text
thesis/
├── README.md
├── AGENTS.md
├── TODO.md
├── pyproject.toml
├── research/
│   ├── control/                         [future] layers and insertion adapters
│   ├── adaptation/                      [future] z optimization
│   ├── policies/prepare_policy.py
│   └── evaluation/
│       ├── aggregate_eval.py
│       ├── verify_eval.py
│       └── provenance.py
├── robotics/
│   ├── sim/                             scenes, controllers, viewer
│   ├── hardware/
│   │   ├── ur5/                         RTDE examples with explicit main()
│   │   └── hx5/                         [future] hand integration
│   ├── teleop/                          Leap, retargeting and replay
│   ├── data/                            [future] collection/conversion
│   └── policies/
│       ├── scripted/
│       ├── rl/                          MJX envs, training and evaluation
│       └── il/                          [future] simple imitation policies
├── experiments/
│   ├── libero/
│   │   ├── run_libero10_smolvla.sh
│   │   ├── baselines/local-baseline.json
│   │   └── control_layers/              [future] comparison configurations
│   └── robotics/                        [future] IL/VLA/hardware experiments
├── platforms/
│   ├── paths.py                         resource and runtime path resolution
│   ├── local/
│   │   ├── config.json
│   │   └── config.local.json.example
│   ├── hpc/
│   │   ├── config.json
│   │   ├── config.local.sh.example
│   │   ├── runtime.sh
│   │   ├── checks/gpu_preflight.py
│   │   └── slurm/smolvla_libero/         setup/assets/eval/array/aggregate
│   └── environments/
│       ├── smolvla_libero/               existing constraints and manifests
│       ├── robotics_sim/                observed laptop simulation versions
│       └── robotics_local/              observed device environment versions
├── assets/mujoco/
│   ├── scene_base.xml
│   ├── hx5_right.xml
│   ├── universal_robots_ur5e/            current model, meshes and license
│   ├── robotis_hand/
│   └── objects/
├── lerobot/                             Git subtree
├── ros2_ws/
│   ├── src/thesis_scene/
│   ├── build/                           [ignored]
│   ├── install/                         [ignored]
│   └── log/                             [ignored]
├── tests/
│   ├── robotics/
│   ├── research/
│   └── platforms/
├── docs/
│   ├── architecture.md
│   ├── usage.md
│   ├── notes.md                         historical working notes
│   ├── lerobot_maintenance.md
│   ├── ASU_Sol_HPC_quick.md
│   ├── ASU_Sol_HPC_reference.md
│   ├── provenance/                      historical patches/migration record
│   └── results/                         curated summaries and reports
├── skills/ASU_Sol_HPC/
├── data/                                [ignored] local datasets
├── out/                                 [ignored] outputs and artifact archive
├── .local-backups/                       [ignored] migration backups
└── old_ros_env_for_reference/            [ignored] independent reference checkout
```

## Boundaries

`research` owns control layers, latent optimization and pretrained-policy adapters.
`robotics` owns the physical system, its simulation and simple scripted/RL/IL
policies. `experiments` owns benchmark protocols and combinations of methods,
models and environments. `platforms` owns execution environments and launchers.

Keep LIBERO-specific task configuration in experiments. Add future real-data VLA
fine-tuning experiments under `experiments/robotics`, reusing LeRobot and research
code. Do not force LIBERO and UR5/HX5 into a common environment API before there
is a concrete integration need. Adapt observations and actions explicitly then.

Keep LeRobot changes in its subtree and separate commits. Preserve ROS workspace
layout; ROS packages may call installed project modules without copying them.
The current RL environment builders remain with their policies during this move.

## Installation, configuration and data

The project uses top-level packages installed editable with `pip install
--no-build-isolation --no-deps -e .`. LeRobot is installed separately. Assets and
configuration are resolved from the editable checkout; distributing a standalone
wheel with bundled robot assets is outside this setup.

`platforms/paths.py` resolves the checkout from its own installed location, or
`THESIS_REPO_ROOT`. `THESIS_PLATFORM` selects `local` (default) or `hpc`.
Tracked `config.json` supplies defaults; ignored `config.local.json` supplies
machine overrides. `THESIS_PLATFORM_CONFIG` selects an explicit override file.
`THESIS_DATA_ROOT`, `THESIS_OUTPUT_ROOT` and `THESIS_UR5_IP` override the relevant
settings. Relative configured paths resolve from the checkout. Explicit CLI
file arguments remain relative to the caller's working directory.

HPC shell jobs source `runtime.sh`, with optional ignored `config.local.sh`
overrides. It exports scratch paths and `THESIS_PYTHON_ENV`, then enters the
checkout. Slurm header paths and resources are submission settings: override
them through `sbatch` flags if needed; shell variables cannot expand in headers.

Local outputs remain in `out/`, datasets in `data/`. HPC data, models, caches,
environments and outputs remain under `/scratch/xzhan742/thesis`. Record versions
separately for simulation/MJX, laptop devices and SmolVLA/LIBERO. Observed version
lists are evidence, not universal installation locks. No dependencies were
upgraded by the move.

Track source, static resources/licenses, configs, scripts, environment records,
docs and skills. Ignore raw recordings, checkpoints, videos, caches, archives,
ordinary logs and build output. Curated metrics/reports may be committed with
their input revisions and artifact locations. Existing tracked history is retained.

LIBERO runs record commit/dirty state, inputs, seed, versions, model references and
output paths before evaluation. A run directory must be new. The historical
baseline manifest is identified as a reference; local pinned runs additionally
record their actual model paths. `eval_info.json` remains the authoritative
rollout result. Important scratch outputs require a separate durable backup.

## Migration mapping

| Previous location | Current location |
| --- | --- |
| `sim/`, `view.py` | `robotics/sim/` |
| `tasks/` | `robotics/policies/scripted/` |
| `rl/` | `robotics/policies/rl/` |
| `teleop/` | `robotics/teleop/`, with UR5 scripts in `robotics/hardware/ur5/` |
| `mujoco/` | `assets/mujoco/` |
| `hpc/smolvla_libero/*.sbatch` | `platforms/hpc/slurm/smolvla_libero/` |
| HPC generic Python tools | `research/policies/`, `research/evaluation/` |
| HPC GPU preflight | `platforms/hpc/checks/` |
| `lerobot/run_libero10_smolvla.sh` | `experiments/libero/run_libero10_smolvla.sh` |
| `USAGE.md`, `NOTES.md`, `LEROBOT_MAINTENANCE.md` | `docs/` |
| `docs/hpc/smolvla/` | `docs/provenance/` and `platforms/environments/` |
| `hpc/smolvla_libero/results/` | curated `docs/results/`, raw `out/archive/` |

Old Python entrypoints are retired. See `usage.md` for replacements. Historical
notes, patches and baseline manifests retain their original paths as evidence.
