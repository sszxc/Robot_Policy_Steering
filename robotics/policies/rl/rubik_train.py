"""Brax PPO on RubikEnv (same settings as rl/cube_train.py). Example: python -m robotics.policies.rl.rubik_train --steps 60_000_000"""
from platforms.paths import output_path
import argparse
import functools
import os
import pickle
import time

import jax
import jax.numpy as jnp
from brax.training.agents.ppo import networks as ppo_networks, train as ppo

from robotics.policies.rl.rubik_env import RubikEnv

# brax 0.14.2 still calls this API, removed in jax 0.11
jax.device_put_replicated = lambda x, devices: jax.tree.map(lambda a: jnp.stack([a] * len(devices)), x)

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=60_000_000)
p.add_argument("--envs", type=int, default=4096)
p.add_argument("--evals", type=int, default=30)
p.add_argument("--out", default=str(output_path('rl/rubik')))
p.add_argument("--assist_prob", type=float, default=0.0, help="fraction of training episodes with a spring fixture")
p.add_argument("--w_vel", type=float, default=0.0, help="cube linear-speed penalty (anti-fling)")
p.add_argument("--w_prog", type=float, default=20.0, help="layer progress reward weight")
p.add_argument("--detent", type=float, default=0.01, help="training detent peak torque (eval env keeps 0.01)")
p.add_argument("--restore", help="params .pkl to warm-start from")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "log.txt"), "a")
t0 = time.time()


def progress(step, m):
    n = max(m["eval/avg_episode_length"], 1)
    line = (f"{time.time() - t0:7.0f}s step {step:>11,d}  reward {m['eval/episode_reward']:9.1f}  "
            f"quarter turns {m['eval/episode_successes']:6.2f}  drop {m['eval/episode_dropped']:.2f}  "
            f"layer_err {m['eval/episode_layer_err'] / n:.2f}  cube_rot {m['eval/episode_cube_rot'] / n:.2f}  len {n:.0f}")
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def save(step, make_policy, params):
    for name in (f"params_{step}.pkl", "params.pkl"):
        with open(os.path.join(a.out, name), "wb") as f:
            pickle.dump(params, f)


restore = pickle.load(open(a.restore, "rb")) if a.restore else None
net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(512, 256, 128),
                        value_hidden_layer_sizes=(512, 256, 128))
ppo.train(
    environment=RubikEnv(assist_prob=a.assist_prob, w_vel=a.w_vel, w_prog=a.w_prog,
                         detent=a.detent), eval_env=RubikEnv(),  # eval env: no fixture
    num_timesteps=a.steps,
    num_evals=a.evals, episode_length=600, num_envs=a.envs, num_eval_envs=128, unroll_length=40,
    batch_size=256, num_minibatches=32, num_updates_per_batch=4, discounting=0.99, learning_rate=3e-4,
    entropy_cost=1e-2, reward_scaling=0.1, normalize_observations=True, max_grad_norm=1.0,
    clipping_epsilon=0.2, network_factory=net, progress_fn=progress, policy_params_fn=save,
    restore_params=restore, seed=0,
)
