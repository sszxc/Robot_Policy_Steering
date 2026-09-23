"""Tiny MJX compile/step check for BaodingEnv: python -m rl.baoding_smoke"""
import time

import jax
import jax.numpy as jnp

from rl.baoding_env import BaodingEnv

env = BaodingEnv()
n = 64
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
s = reset(jax.random.split(jax.random.PRNGKey(0), n))
s = step(s, jnp.zeros((n, env.action_size)))
jax.block_until_ready(s.obs)
t = time.time()
for _ in range(20):
    s = step(s, 0.3 * jax.random.normal(jax.random.PRNGKey(1), (n, env.action_size)))
jax.block_until_ready(s.obs)
dt = time.time() - t
print("obs", s.obs.shape, "finite", bool(jnp.isfinite(s.obs).all()), "reward mean", float(s.reward.mean()),
      "dropped", float(s.done.mean()), f"env steps/s {20 * n / dt:.0f}")
