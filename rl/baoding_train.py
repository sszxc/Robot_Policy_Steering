"""Brax PPO on BaodingEnv. Example: XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python -m rl.baoding_train --steps 80_000_000"""
import argparse
import functools
import os
import pickle
import time

import jax
import jax.numpy as jnp
from brax.training.agents.ppo import networks as ppo_networks, train as ppo

from rl.baoding_env import BaodingEnv

# brax 0.14.2 still calls this API, removed in jax 0.11
jax.device_put_replicated = lambda x, devices: jax.tree.map(lambda a: jnp.stack([a] * len(devices)), x)

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=80_000_000)
p.add_argument("--period", type=float, default=8.0, help="seconds per full orbit of the targets")
p.add_argument("--w_turn", type=float, default=10.0, help="reward weight on pair-axis rotation progress")
p.add_argument("--envs", type=int, default=2048)
p.add_argument("--evals", type=int, default=25)
p.add_argument("--out", default="out/rl/baoding")
p.add_argument("--restore", help="params .pkl to warm-start from")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "log.txt"), "a")
t0 = time.time()


def progress(step, m):
    n = max(m["eval/avg_episode_length"], 1)
    line = (f"{time.time() - t0:7.0f}s step {step:>11,d}  reward {m['eval/episode_reward']:9.1f}  "
            f"turn {m['eval/episode_turn']:6.2f} rad  track_err {1000 * m['eval/episode_track_err'] / n:5.1f} mm  "
            f"drop {m['eval/episode_dropped']:.2f}  len {n:.0f}")
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
    environment=BaodingEnv(a.period, w_turn=a.w_turn), eval_env=BaodingEnv(a.period, w_turn=a.w_turn), num_timesteps=a.steps,
    num_evals=a.evals, episode_length=400, num_envs=a.envs, num_eval_envs=128, unroll_length=40,
    batch_size=256, num_minibatches=16, num_updates_per_batch=4, discounting=0.99, learning_rate=3e-4,
    entropy_cost=1e-2, reward_scaling=0.1, normalize_observations=True, max_grad_norm=1.0,
    clipping_epsilon=0.2, network_factory=net, progress_fn=progress, policy_params_fn=save,
    restore_params=restore, seed=0,
)
