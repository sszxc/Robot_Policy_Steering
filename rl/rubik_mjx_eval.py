"""Batched MJX rollout of Rubik checkpoints (same env as training): quarter turns/ep and drop rate.
Example: python -m rl.rubik_mjx_eval --params out/rl/rubik/params_*.pkl"""
import argparse
import pickle

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from rl.rubik_env import RubikEnv

p = argparse.ArgumentParser()
p.add_argument("--params", nargs="+", default=["out/rl/rubik/best_params.pkl"])
p.add_argument("--episodes", type=int, default=64)
p.add_argument("--assist_prob", type=float, default=0.0, help="diagnostic: episodes with the spring fixture")
a = p.parse_args()

env = RubikEnv(assist_prob=a.assist_prob)
net = ppo_networks.make_ppo_networks(76, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128))
make = ppo_networks.make_inference_fn(net)


@jax.jit
def rollout(params):
    policy = jax.vmap(make(params, deterministic=True))
    s = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), a.episodes))
    keys = jax.random.split(jax.random.PRNGKey(1), a.episodes)

    def body(c, _):  # stop counting after the first done
        s, turns, done, steps = c
        s = jax.vmap(env.step)(s, policy(s.obs, keys)[0])
        turns = turns + s.metrics["successes"] * (1 - done)
        steps = steps + (1 - done)
        return (s, turns, jax.numpy.maximum(done, s.done), steps), None

    z = jax.numpy.zeros(a.episodes)
    (s, turns, done, steps), _ = jax.lax.scan(body, (s, z, z, z), None, length=600)
    return turns, done, steps, s.info["sign"]


for path in a.params:
    turns, done, steps, sign = map(np.asarray, rollout(pickle.load(open(path, "rb"))))
    per_dir = "  ".join(f"dir {g:+d}: {turns[sign == g].mean():.2f}" for g in (1, -1))
    print(f"{path}: quarter turns/ep {turns.mean():.2f} ({per_dir}), drop {done.mean():.2f}, "
          f"mean len {steps.mean():.0f}, eps with >=1 turn {(turns > 0).mean():.2f}", flush=True)
