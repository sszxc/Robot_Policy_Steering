"""DAgger: distill the noisy privileged Baoding teacher into a top-down RGB student (2 frames + hand proprio).
Labels are the teacher's deterministic actions on every visited state; aux loss on ball positions.
Example: XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python -m robotics.policies.rl.baoding_dagger --teacher out/rl/baoding_noisy/teacher/best_params.pkl"""
from platforms.paths import output_path
import argparse
import os
import pickle
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from robotics.policies.rl.baoding_noisy_env import BaodingNoisyEnv
from robotics.policies.rl.baoding_noisy_train import load_policy

T = 400  # steps per episode


class Student(nn.Module):
    """NatureCNN (same as the vision-PPO actor) -> concat proprio -> MLP; heads: action (tanh) and ball positions."""
    @nn.compact
    def __call__(self, pix, prop):
        x = pix.astype(jnp.float32) / 255.0 - 0.5
        for c, k, s in ((32, 8, 4), (64, 4, 2), (64, 3, 1)):
            x = nn.relu(nn.Conv(c, (k, k), (s, s))(x))
        h = jnp.concatenate([x.reshape(x.shape[:-3] + (-1,)), prop], -1)
        for w in (512, 256, 128):
            h = nn.swish(nn.Dense(w)(h))
        return jnp.tanh(nn.Dense(20)(h)), nn.Dense(6)(h)


def make_student_act(params, norm):
    model = Student()
    return jax.jit(lambda obs: model.apply(params, obs["pixels/view_0"], (obs["vis_state"] - norm[0]) / norm[1])[0])


def load_student(path):
    d = pickle.load(open(path, "rb"))
    return make_student_act(d["params"], d["norm"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=str(output_path("rl/baoding_noisy/teacher/best_params.pkl")))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--envs", type=int, default=192)
    p.add_argument("--grad_steps", type=int, default=3000, help="per iteration")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--w_aux", type=float, default=0.1)
    p.add_argument("--out", default=str(output_path("rl/baoding_noisy/dagger")))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "a")

    env = BaodingNoisyEnv(render=True)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    teacher = load_policy(a.teacher, False, env)
    model = Student()

    # aggregated dataset (valid steps only); prev-frame index gives the 2-frame stack without storing it twice
    M = a.iters * a.envs * T
    frames = np.empty((M, 64, 64, 3), np.uint8)
    prev = np.empty(M, np.int64)
    prop = np.empty((M, 66), np.float32)
    label = np.empty((M, 20), np.float32)
    ball = np.empty((M, 6), np.float32)
    n = 0

    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 64, 64, 6), jnp.uint8), jnp.zeros((1, 66)))
    opt = optax.adam(3e-4)
    opt_state = opt.init(params)
    norm = None

    @jax.jit
    def train_step(params, opt_state, pix, x, y, yb):
        def loss_fn(p):
            act, bp = model.apply(p, pix, x)
            la, lb = jnp.mean((act - y) ** 2), jnp.mean((bp - yb) ** 2)
            return la + a.w_aux * lb, (la, lb)
        (_, (la, lb)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(params, upd), opt_state, la, lb

    t0 = time.time()
    for it in range(a.iters):
        beta = 1.0 if it == 0 else 0.0  # iteration 0 = teacher rollouts (behavior cloning), then student-driven
        student = make_student_act(params, norm) if it > 0 else None
        s = reset(jax.random.split(jax.random.PRNGKey(1000 + it), a.envs))
        alive = np.ones(a.envs, bool)
        last = np.full(a.envs, -1)
        turn, err = np.zeros(a.envs), []
        for t in range(T):
            obs = s.obs
            y = teacher(obs)
            act = y if beta == 1.0 else student(obs)
            k = np.flatnonzero(alive)
            idx = n + np.arange(len(k))
            frames[idx] = np.asarray(obs["pixels/view_0"][..., :3])[k]
            prev[idx] = np.where(last[k] >= 0, last[k], idx)
            prop[idx] = np.asarray(obs["vis_state"])[k]
            label[idx] = np.asarray(y)[k]
            ball[idx] = np.asarray(obs["privileged_state"][:, 60:66])[k]
            last[k] = idx
            n += len(k)
            s = step(s, act)
            turn += np.asarray(s.metrics["turn"]) * alive
            err.append(np.where(alive, np.asarray(s.metrics["track_err"]), np.nan))
            alive &= np.asarray(s.done) == 0
        roll = (f"iter {it} ({'teacher' if beta else 'student'} rollouts): pair turn {turn.mean():.2f} rad/ep, "
                f"drop {1 - alive.mean():.2f}, track_err {1000 * np.nanmean(err):.1f} mm, data {n:,}")
        if norm is None:
            norm = (jnp.array(prop[:n].mean(0)), jnp.array(prop[:n].std(0) + 1e-3))
        rng = np.random.default_rng(it)
        for g in range(a.grad_steps):
            i = rng.integers(0, n, a.batch)
            pix = jnp.array(np.concatenate([frames[i], frames[prev[i]]], -1))
            params, opt_state, la, lb = train_step(params, opt_state, pix, (jnp.array(prop[i]) - norm[0]) / norm[1],
                                                   jnp.array(label[i]), jnp.array(ball[i]))
        line = f"{time.time() - t0:6.0f}s {roll}; train action mse {float(la):.4f}, ball mse {float(lb):.4f}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        with open(os.path.join(a.out, f"student_{it}.pkl"), "wb") as f:
            pickle.dump({"params": params, "norm": norm}, f)
        with open(os.path.join(a.out, "student.pkl"), "wb") as f:
            pickle.dump({"params": params, "norm": norm}, f)
