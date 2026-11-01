"""Closed-loop check for a privileged Baoding checkpoint (MJX): normal vs ball obs frozen at reset vs a mid-episode push.
Example: python -m robotics.policies.rl.baoding_check --params out/rl/baoding/best_params.pkl out/rl/baoding/pace_params.pkl"""
from platforms.paths import output_path
import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from robotics.policies.rl import baoding_env as E

p = argparse.ArgumentParser()
p.add_argument("--params", nargs="+", default=[str(output_path('rl/baoding/best_params.pkl'))])
p.add_argument("--period", type=float, default=16.0)
p.add_argument("--episodes", type=int, default=64)
a = p.parse_args()

env = E.BaodingEnv(a.period)
net = ppo_networks.make_ppo_networks(78, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128))
make = ppo_networks.make_inference_fn(net)
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
BALL = slice(60, 72)  # ball position + velocity part of the obs


def run(policy, mode):
    s = reset(jax.random.split(jax.random.PRNGKey(0), a.episodes))
    frozen = s.obs[:, BALL]
    keys = jax.random.split(jax.random.PRNGKey(1), a.episodes)
    push = jax.random.normal(jax.random.PRNGKey(2), (a.episodes, 2)) * 0.15   # in-plane (y, z) m/s kick on ball0
    turn, err, done = np.zeros(a.episodes), [], np.zeros(a.episodes, bool)
    for t in range(400):
        obs = s.obs.at[:, BALL].set(frozen) if mode == "frozen" else s.obs
        act, _ = policy(obs, keys)
        if mode == "push" and t == 200:
            v = jnp.concatenate([jnp.zeros((a.episodes, 1)), push], 1) @ env.R.T
            s = s.replace(pipeline_state=s.pipeline_state.replace(qvel=s.pipeline_state.qvel.at[:, 20:23].add(v)))
        s = step(s, act)
        alive = ~done
        turn += np.asarray(s.metrics["turn"]) * alive
        err.append(np.where(alive, np.asarray(s.metrics["track_err"]), np.nan))
        done |= np.asarray(s.done) > 0
    return turn.mean(), done.mean(), 1000 * np.nanmean(err)


for path in a.params:
    policy = jax.jit(jax.vmap(make(pickle.load(open(path, "rb")), deterministic=True)))
    for mode in ("normal", "frozen", "push"):
        turn, drop, err = run(policy, mode)
        print(f"{path} [{mode:6s}] pair turn {turn:6.1f} rad/ep  drop {drop:.2f}  track_err {err:5.1f} mm", flush=True)
