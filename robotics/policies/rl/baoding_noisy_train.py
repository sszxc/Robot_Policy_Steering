"""Brax PPO on BaodingNoisyEnv: noisy privileged teacher (default) or direct vision policy (--vision; critic stays
privileged). Example: XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 python -m robotics.policies.rl.baoding_noisy_train --steps 50_000_000
v2 (speed command + MuJoCo Warp): add --periods 2 16 --warp"""
from platforms.paths import output_path
import argparse
import functools
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks, networks_vision, train as ppo

from robotics.policies.rl.baoding_noisy_env import BaodingNoisyEnv, add_task_args, task_kw

# brax 0.14.2 still calls this API, removed in jax 0.11
jax.device_put_replicated = lambda x, devices: jax.tree.map(lambda a: jnp.stack([a] * len(devices)), x)


def make_net(vision):
    """Network factory shared by train / eval / DAgger (teacher)."""
    if vision:
        return functools.partial(networks_vision.make_ppo_networks_vision, policy_obs_key="vis_state",
                                 value_obs_key="privileged_state", normalise_channels=True, cnn_global_pool="none",
                                 policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128))
    return functools.partial(ppo_networks.make_ppo_networks, policy_obs_key="state", value_obs_key="privileged_state",
                             policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128))


def load_policy(path, vision, env):
    """Deterministic batched brax policy: obs dict (N, ...) -> actions (N, 20)."""
    shapes = {k: (v,) if isinstance(v, int) else v for k, v in env.observation_size.items()}
    net = make_net(vision)(shapes, 20, preprocess_observations_fn=running_statistics.normalize)
    pol = ppo_networks.make_inference_fn(net)(pickle.load(open(path, "rb")), deterministic=True)
    return jax.jit(lambda obs: pol(obs, jax.random.PRNGKey(0))[0])


def pad_critic(params, n, at):
    """Warm-start a --dr run from a non-DR checkpoint: insert n critic inputs (zero weights) before the last `at`."""
    norm, pol, val = params
    ins = lambda x, v: jnp.concatenate([x[:-at], jnp.full((n,) + x.shape[1:], v, x.dtype), x[-at:]])
    k = "privileged_state"
    cnt = float(norm.count.lo) + 2.0 ** 32 * float(norm.count.hi)
    norm = norm.replace(mean={**norm.mean, k: ins(norm.mean[k], 0.0)}, std={**norm.std, k: ins(norm.std[k], 0.2)},
                        summed_variance={**norm.summed_variance, k: ins(norm.summed_variance[k], 0.04 * cnt)})
    h0 = val["params"]["hidden_0"]
    return norm, pol, {"params": {**val["params"], "hidden_0": {**h0, "kernel": ins(h0["kernel"], 0.0)}}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50_000_000)
    p.add_argument("--vision", action="store_true")
    p.add_argument("--period", type=float, default=16.0)
    p.add_argument("--w_turn", type=float, default=50.0)
    p.add_argument("--periods", type=float, nargs=2, help="v2 speed command: period range (s), log-uniform")
    p.add_argument("--warp", action="store_true", help="MJX-Warp physics + mesh rendering")
    p.add_argument("--cam", default="top", choices=["top", "wrist"], help="--warp camera (vision)")
    p.add_argument("--tilt", type=float, help="max palm tilt (deg) from palm-up, any direction; adds gravity obs")
    p.add_argument("--tilt_about", default="up", choices=["up", "nominal"], help="tilt cap axis")
    p.add_argument("--dr", action="store_true", help="physics domain randomization (+0..2 step action delay)")
    p.add_argument("--w_over", type=float, default=0.0, help="penalty weight (x w_turn) on rotation beyond the pace")
    p.add_argument("--envs", type=int, help="default 2048 (state) / 1024 (vision)")
    p.add_argument("--evals", type=int, default=25)
    p.add_argument("--out", help="default out/rl/baoding_{noisy,cmd}/{teacher,vision_rl}")
    p.add_argument("--rel", action="store_true", help="no orbit tracking: pair-rotation reward, wide start")
    p.add_argument("--pert", nargs="+", default=[], help="extra perturbations, key=value (PERT in baoding_noisy_env)")
    p.add_argument("--restore", help="params .pkl to warm-start from")
    add_task_args(p)
    a = p.parse_args()
    out = a.out or str(output_path(f"rl/baoding_{'cmd' if a.periods else 'noisy'}/" + ("vision_rl" if a.vision else "teacher")))
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")
    json.dump(task_kw(a), open(os.path.join(out, "task.json"), "w"))  # task options of this teacher
    t0 = time.time()

    def progress(step, m):
        n = max(m["eval/avg_episode_length"], 1)
        line = (f"{time.time() - t0:7.0f}s step {step:>11,d}  reward {m['eval/episode_reward']:9.1f}  "
                f"turn {m['eval/episode_turn']:6.2f} rad  track_err {1000 * m['eval/episode_track_err'] / n:5.1f} mm  "
                f"pace {100 * m['eval/episode_pace'] / n:4.0f}%  drop {m['eval/episode_dropped']:.2f}  len {n:.0f}")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def save(step, make_policy, params):
        for name in (f"params_{step}.pkl", "params.pkl"):
            with open(os.path.join(out, name), "wb") as f:
                pickle.dump(params, f)

    # vision: smaller rollout batch (pixels dominate memory: 1024 envs x 20 steps x 2 x 24.6 KB)
    kw = (dict(num_envs=a.envs or 1024, batch_size=128, num_minibatches=8, unroll_length=20, num_eval_envs=128)
          if a.vision else dict(num_envs=a.envs or 2048, batch_size=256, num_minibatches=16, unroll_length=40,
                                num_eval_envs=128))
    tilt = None if a.tilt is None else np.radians(a.tilt)
    restore = pickle.load(open(a.restore, "rb")) if a.restore else None
    if a.dr and restore is not None and restore[2]["params"]["hidden_0"]["kernel"].shape[0] == 85 + (a.periods is not None) + 3 * (tilt is not None):
        restore = pad_critic(restore, 7, (a.periods is not None) + 3 * (tilt is not None))
    env = lambda n: BaodingNoisyEnv(a.period, a.w_turn, render=a.vision, periods=a.periods, warp=a.warp, nworld=n,
                                    w_over=a.w_over, cam=a.cam, tilt_max=tilt, dr=a.dr, tilt_about=a.tilt_about, rel=a.rel,
                                    pert={k: float(v) for k, v in (x.split("=") for x in a.pert)}, **task_kw(a))
    ppo.train(
        environment=env(kw["num_envs"]), eval_env=env(kw["num_eval_envs"]), num_timesteps=a.steps, num_evals=a.evals, episode_length=400,
        num_updates_per_batch=4, discounting=0.99, learning_rate=3e-4, entropy_cost=1e-2, reward_scaling=0.1,
        normalize_observations=True, max_grad_norm=1.0, clipping_epsilon=0.2, network_factory=make_net(a.vision),
        progress_fn=progress, policy_params_fn=save, vision=a.vision, seed=0,
        restore_params=restore, **kw,
    )
