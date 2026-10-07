"""Load SmolVLA and a LIBERO vector env once; run batched episodes from chosen init states."""
import json
import multiprocessing as mp
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.random_utils import set_seed

from platforms.paths import output_path
from research.evaluation.provenance import record


def add_args(parser):
    parser.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2, help="parallel envs")
    parser.add_argument("--episode-length", type=int, default=None, help="default: suite max")
    parser.add_argument("--sync-envs", action="store_true", help="SyncVectorEnv (debugging)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)


def new_run_dir(args, name):
    out = args.output_dir or output_path(f"{name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "provenance.json").write_text(json.dumps(record(vars(args)), indent=2, default=str) + "\n")
    return out


def load_policy(args, env_cfg):
    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path = Path(args.policy)
    cfg.device = "cuda"
    policy = make_policy(cfg=cfg, env_cfg=env_cfg).eval()
    policy.model.sample_noise = _paired_noise
    return policy, cfg


def load(args):
    """Returns (policy, env, procs); procs = (env_pre, env_post, pre, post) as rollout() expects."""
    set_seed(args.seed)
    env_cfg = env_config(args)
    # Workers share imports via forkserver preload and return freed sim memory: ~1.4 vs 2.2 GB/env, same results
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", str(64 << 20))
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    mp.set_forkserver_preload(["lerobot.envs.libero"])
    env = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=not args.sync_envs)
    env = env[args.suite][args.task_id]
    if hasattr(env, "_ensure"):  # lerobot's lazy async wrapper has no set_attr
        env._ensure()
        env = env._env
    policy, cfg = load_policy(args, env_cfg)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.policy,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return policy, env, (env_pre, env_post, pre, post)


def env_config(args):
    return LiberoEnv(task=args.suite, task_ids=[args.task_id], episode_length=args.episode_length,
                     control_mode="relative", init_states=True, hard_reset=True)


_GENS = []  # one generator per env: noise depends only on its seed and the step, not the batch


def _paired_noise(shape, device):
    assert len(_GENS) == shape[0], "run_batch sets one noise seed per env"
    return torch.stack([torch.randn(shape[1:], generator=g, device=device) for g in _GENS])


def run_batch(policy, env, procs, init_ids, seed, noise_seeds=None, video_envs=(), obs_every=0):
    """Sub-env i starts from init state init_ids[i] with flow noise seeded by noise_seeds[i]
    (default seed + init id). Returns success[B], steps[B], action[B, T, A], state[B, T, 8]
    (raw eef pos, axis-angle, gripper qpos; before each action), frames {env index: [T, H, W, 3]} for video_envs and, if
    obs_every, policy inputs every obs_every steps as a list of per-sample dicts."""
    init_ids = [int(i) for i in init_ids]
    noise_seeds = [seed + i for i in init_ids] if noise_seeds is None else noise_seeds
    _GENS[:] = [torch.Generator("cuda").manual_seed(int(s)) for s in noise_seeds]
    env.set_attr("init_state_id", init_ids)
    env_pre, env_post, pre, post = procs
    states, inputs = [], []

    def env_pre_rec(obs):
        obs = env_pre(obs)
        states.append(obs["observation.state"].float().cpu())
        return obs

    select = policy.select_action
    if obs_every:
        def record_input(batch):
            if (len(states) - 1) % obs_every == 0:
                inputs.append((len(states) - 1, {k: v.cpu() for k, v in batch.items() if torch.is_tensor(v)}
                               | {"task": batch["task"]}))
            return select(batch)
        policy.select_action = record_input
    video_envs = list(video_envs)
    frames = []
    callback = (lambda e: frames.append(np.stack(e.call("render"))[video_envs])) if video_envs else None
    try:
        data = rollout(env, policy, env_pre_rec, env_post, pre, post,
                       seeds=[seed + i for i in init_ids], render_callback=callback)
    finally:
        policy.select_action = select
    done_idx = data["done"].int().argmax(1)
    valid = torch.arange(data["done"].shape[1])[None] <= done_idx[:, None]
    steps = (done_idx + 1).numpy()
    frames = np.stack(frames, 1) if video_envs else []
    samples = [{k: v[i] for k, v in b.items()}
               for t, b in inputs for i in range(len(init_ids)) if t < steps[i]]
    return {
        "success": (data["success"].bool() & valid).any(1).numpy(),
        "steps": steps,
        "action": data["action"].numpy(),
        "state": torch.stack(states, 1).numpy(),
        "frames": {i: f[:steps[i] + 1] for i, f in zip(video_envs, frames)},
        "inputs": samples,
    }
