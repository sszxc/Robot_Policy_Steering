"""Batched MJX rollout of a pen checkpoint (same env as training), per-direction stats.
Example: python -m robotics.policies.rl.pen_mjx_eval --params out/rl/pen/params.pkl"""
from platforms.paths import output_path
import argparse
import pickle

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from robotics.policies.rl import pen_env as E

p = argparse.ArgumentParser()
p.add_argument("--params", nargs="+", default=[str(output_path('rl/pen/best_params.pkl'))])
p.add_argument("--episodes", type=int, default=64)
a = p.parse_args()

env = E.PenEnv()
net = ppo_networks.make_ppo_networks(73, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128))
make = ppo_networks.make_inference_fn(net)
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

for path in a.params:
    policy = jax.jit(jax.vmap(make(pickle.load(open(path, "rb")), deterministic=True)))
    s = reset(jax.random.split(jax.random.PRNGKey(0), a.episodes))
    keys = jax.random.split(jax.random.PRNGKey(1), a.episodes)
    turn, done, tilt = np.zeros(a.episodes), np.zeros(a.episodes, bool), []
    for t in range(400):
        act, _ = policy(s.obs, keys)
        s = step(s, act)
        turn += np.asarray(s.metrics["turn"]) * ~done
        done |= np.asarray(s.done) > 0
        tilt.append(np.abs(np.asarray(jax.vmap(E.pen_axis)(s.pipeline_state.qpos[:, 23:27]))[:, 2]))
    d = np.asarray(s.info["dir"])
    for sgn in (1, -1):
        k = d == sgn
        print(f"{path} dir {sgn:+d}: {turn[k].mean() / (2 * np.pi):.2f} rev/ep ({turn[k].mean() / 20:.2f} rad/s over 20 s), "
              f"drop rate {done[k].mean():.2f}, n={k.sum()}")
    print(f"  max |axis z| {np.max(tilt):.2f} (flat pen = 0)", flush=True)
