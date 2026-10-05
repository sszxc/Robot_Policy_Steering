"""Load SmolVLA and a LIBERO vector env once; run batched episodes from chosen init states."""
import json
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
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=9)
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


def load(args):
    """Returns (policy, env, procs); procs = (env_pre, env_post, pre, post) as rollout() expects."""
    set_seed(args.seed)
    env_cfg = LiberoEnv(task=args.suite, task_ids=[args.task_id], episode_length=args.episode_length,
                        control_mode="relative", init_states=True, hard_reset=True)
    env = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=not args.sync_envs)
    env = env[args.suite][args.task_id]
    if hasattr(env, "_ensure"):  # lerobot's lazy async wrapper has no set_attr
        env._ensure()
        env = env._env
    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path = Path(args.policy)
    cfg.device = "cuda"
    policy = make_policy(cfg=cfg, env_cfg=env_cfg).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.policy,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return policy, env, (env_pre, env_post, pre, post)


def run_batch(policy, env, procs, init_ids, seed, video_envs=()):
    """Sub-env i starts from init state init_ids[i]; frames are kept only for video_envs.
    Returns success[B], steps[B], action[B, T, A] and frames {env index: [T, H, W, 3]}."""
    init_ids = [int(i) for i in init_ids]
    env.set_attr("init_state_id", init_ids)
    video_envs = list(video_envs)
    frames = []
    callback = (lambda e: frames.append(np.stack(e.call("render"))[video_envs])) if video_envs else None
    data = rollout(env, policy, *procs, seeds=[seed + i for i in init_ids], render_callback=callback)
    done_idx = data["done"].int().argmax(1)
    valid = torch.arange(data["done"].shape[1])[None] <= done_idx[:, None]
    steps = (done_idx + 1).numpy()
    frames = np.stack(frames, 1) if video_envs else []
    return {
        "success": (data["success"].bool() & valid).any(1).numpy(),
        "steps": steps,
        "action": data["action"],
        "frames": {i: f[:steps[i] + 1] for i, f in zip(video_envs, frames)},
    }
