"""Batched MJX rollout of a Baoding checkpoint with orbit diagnostics.
Example: XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python -m robotics.policies.rl.baoding_mjx_eval --params out/rl/baoding/params.pkl"""
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


def orbit_stats(b, alive):
    """b: (T, N, 2, 3) hand-frame ball centers. Per-episode pair turn, each ball's turn about the
    pair midpoint, yz separation, and ball height above the palm."""
    yz = b[..., 1:]
    mid = yz.mean(2, keepdims=True)
    ang = np.arctan2(yz[..., 1] - mid[..., 1], yz[..., 0] - mid[..., 0])       # (T, N, 2)
    d = (np.diff(ang, axis=0) + np.pi) % (2 * np.pi) - np.pi
    d *= alive[1:, :, None]
    pair = np.arctan2(yz[:, :, 1, 1] - yz[:, :, 0, 1], yz[:, :, 1, 0] - yz[:, :, 0, 0])
    dp = ((np.diff(pair, axis=0) + np.pi) % (2 * np.pi) - np.pi) * alive[1:]
    sep = np.linalg.norm(yz[:, :, 0] - yz[:, :, 1], axis=-1)
    return dp.sum(0), d.sum(0), sep, b[..., 0] - E.PALM_X - E.B.R


for path in a.params:
    policy = jax.jit(jax.vmap(make(pickle.load(open(path, "rb")), deterministic=True)))
    s = reset(jax.random.split(jax.random.PRNGKey(0), a.episodes))
    keys = jax.random.split(jax.random.PRNGKey(1), a.episodes)
    B, alive, done = [], [], np.zeros(a.episodes, bool)
    for t in range(400):
        act, _ = policy(s.obs, keys)
        s = step(s, act)
        done |= np.asarray(s.done) > 0
        B.append(np.asarray(jax.vmap(E.balls_hand, (0, None, None))(s.pipeline_state.qpos, env.R, env.o)))
        alive.append(~done)
    B, alive = np.array(B), np.array(alive, float)
    turn, ball_turn, sep, lift = orbit_stats(B, alive)
    live = alive > 0
    print(f"{path}: pair turn {turn.mean():.1f} rad/ep (median {np.median(turn):.1f}), drop rate {done.mean():.2f}, "
          f"ball turns about midpoint {ball_turn.mean(0).round(1)}, yz separation mean {1000 * sep[live].mean():.1f} mm "
          f"(min {1000 * sep[live].min():.1f}), max lift above palm contact {1000 * lift[live].max():.1f} mm", flush=True)
