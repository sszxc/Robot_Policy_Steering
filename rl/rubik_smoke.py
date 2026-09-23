"""MJX smoke test for RubikEnv: layout check, then batched reset/step throughput with random actions.
python -m rl.rubik_smoke [--envs 512] [--steps 50]"""
import argparse
import time

import jax
import jax.numpy as jnp

from rl.rubik_env import RubikEnv

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=512)
p.add_argument("--steps", type=int, default=50)
a = p.parse_args()

env = RubikEnv()
m = env.mj_model
print("nq", m.nq, "nv", m.nv, "nu", m.nu, "hinge qpos adr", m.joint("rubik_top").qposadr[0],
      "dof", m.joint("rubik_top").dofadr[0], "cube qpos adr", m.joint("rubik").qposadr[0])
assert m.joint("rubik_top").qposadr[0] == 27 and m.joint("rubik_top").dofadr[0] == 26

reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
state = reset(jax.random.split(jax.random.PRNGKey(0), a.envs))
key = jax.random.PRNGKey(1)
t0 = time.time()
state = step(state, jnp.zeros((a.envs, env.action_size)))
jax.block_until_ready(state.obs)
print(f"compile+first step {time.time() - t0:.1f}s, obs {state.obs.shape}")
t0 = time.time()
for _ in range(a.steps):
    key, k = jax.random.split(key)
    state = step(state, jax.random.uniform(k, (a.envs, env.action_size), minval=-1, maxval=1))
jax.block_until_ready(state.obs)
dt = time.time() - t0
print(f"{a.envs * a.steps / dt:,.0f} env-steps/s ({a.envs * a.steps * env.n_frames / dt:,.0f} physics steps/s)")
print("nan:", bool(jnp.isnan(state.obs).any()), " dropped frac:", float(state.metrics["dropped"].mean()),
      " mean |layer err| deg:", float(jnp.degrees(state.metrics["layer_err"]).mean()),
      " reward mean:", float(state.reward.mean()))
