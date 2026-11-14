"""DAgger: distill the noisy privileged Baoding teacher into a top-down RGB student (2 frames + hand proprio).
Labels are the teacher's deterministic actions on every visited state; aux loss on ball positions.
Example: XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python -m robotics.policies.rl.baoding_dagger --teacher out/rl/baoding_noisy/teacher/best_params.pkl
v2 (speed command + MuJoCo Warp): --periods 2 16 --warp --teacher out/rl/baoding_cmd/teacher/best_params.pkl"""
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

from robotics.policies.rl.baoding_noisy_env import BaodingNoisyEnv, add_task_args, task_kw
from robotics.policies.rl.baoding_noisy_train import load_policy

T = 400  # steps per episode


def keypoints(x, temp):
    """Spatial softmax: (..., H, W, C) maps -> (..., 2C) expected (x, y) image coordinates in [-1, 1]."""
    h, w, c = x.shape[-3:]
    p = jax.nn.softmax(x.reshape(x.shape[:-3] + (h * w, c)) / temp, axis=-2)
    gy, gx = jnp.meshgrid(jnp.linspace(-1, 1, h), jnp.linspace(-1, 1, w), indexing="ij")
    return jnp.concatenate([jnp.einsum("...nc,n->...c", p, gx.ravel()), jnp.einsum("...nc,n->...c", p, gy.ravel())], -1)


class Student(nn.Module):
    """Encoder -> concat proprio -> MLP -> action (tanh); aux head: ball positions from the image features alone
    (no proprio: also the probe for real images).
    enc='flat': NatureCNN on the frame stack, flattened map. enc='ssm': per-frame CNN (shared, half-resolution maps)
    -> spatial softmax keypoints (32 per frame), all frames concatenated."""
    enc: str = "flat"

    @nn.compact
    def __call__(self, pix, prop):
        x = pix.astype(jnp.float32) / 255.0 - 0.5
        if self.enc == "flat":
            for c, k, s in ((32, 8, 4), (64, 4, 2), (64, 3, 1)):
                x = nn.relu(nn.Conv(c, (k, k), (s, s))(x))
            f = x.reshape(x.shape[:-3] + (-1,))
        else:
            nf = x.shape[-1] // 3
            x = jnp.concatenate(jnp.split(x, nf, -1), 0)  # frames along the batch axis, shared weights
            for c, k, s in ((32, 5, 2), (64, 3, 1), (64, 3, 1)):
                x = nn.relu(nn.Conv(c, (k, k), (s, s))(x))
            temp = jnp.exp(self.param("log_temp", nn.initializers.zeros, ()))
            f = keypoints(nn.Conv(32, (1, 1))(x), temp)
            f = jnp.concatenate(jnp.split(f, nf, 0), -1)
        ball_img = nn.Dense(6)(nn.swish(nn.Dense(128)(f)))
        h = jnp.concatenate([f, prop], -1)
        for w in (512, 256, 128):
            h = nn.swish(nn.Dense(w)(h))
        return jnp.tanh(nn.Dense(20)(h)), ball_img


def make_student_act(params, norm, enc="flat"):
    model = Student(enc)
    return jax.jit(lambda obs: model.apply(params, obs["pixels/view_0"], (obs["vis_state"] - norm[0]) / norm[1])[0])


def load_student(path):
    d = pickle.load(open(path, "rb"))
    return make_student_act(d["params"], d["norm"], d.get("enc", "flat"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--periods", type=float, nargs=2, help="v2 speed command: period range (s)")
    p.add_argument("--warp", action="store_true", help="MJX-Warp physics + mesh rendering")
    p.add_argument("--cam", default="top", choices=["top", "wrist"], help="--warp camera")
    p.add_argument("--tilt", type=float, help="max palm tilt (deg); teacher must be trained with the same flag")
    p.add_argument("--tilt_about", default="up", choices=["up", "nominal"])
    p.add_argument("--dr", action="store_true", help="physics domain randomization")
    p.add_argument("--vis_dr", action="store_true", help="camera-mount / color / photometric randomization")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--envs", type=int, default=192)
    p.add_argument("--grad_steps", type=int, default=3000, help="per iteration")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--w_aux", type=float, default=0.1)
    p.add_argument("--bg", help="--vis_dr background library (.npy from baoding_real --bg)")
    p.add_argument("--stack", type=int, default=2, help="stacked frames")
    p.add_argument("--enc", default="flat", choices=["flat", "ssm"], help="flatten CNN or spatial-softmax keypoints")
    p.add_argument("--rel", action="store_true", help="teacher trained with --rel")
    p.add_argument("--pert", nargs="+", default=[], help="extra perturbations, key=value (PERT in baoding_noisy_env)")
    p.add_argument("--out")
    add_task_args(p)
    a = p.parse_args()
    a.out = a.out or str(output_path(f"rl/baoding_{'cmd' if a.periods else 'noisy'}/dagger"))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "a")

    env = BaodingNoisyEnv(render=True, periods=a.periods, warp=a.warp, nworld=a.envs, cam=a.cam,
                          tilt_max=None if a.tilt is None else np.radians(a.tilt), dr=a.dr,
                          tilt_about=a.tilt_about, vis_dr=a.vis_dr, bg=a.bg, stack=a.stack, rel=a.rel,
                          pert={k: float(v) for k, v in (x.split("=") for x in a.pert)}, **task_kw(a))
    nv = env.observation_size["vis_state"]
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    teacher = load_policy(a.teacher, False, env)
    model = Student(a.enc)

    # aggregated dataset (valid steps only); prev-frame indices give the frame stack without storing it twice
    M = a.iters * a.envs * T
    h, w = env.observation_size["pixels/view_0"][:2]
    frames = np.empty((M, h, w, 3), np.uint8)
    prev = np.empty(M, np.int64)
    prop = np.empty((M, nv), np.float32)
    label = np.empty((M, 20), np.float32)
    ball = np.empty((M, 6), np.float32)
    n = 0

    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, h, w, 3 * a.stack), jnp.uint8), jnp.zeros((1, nv)))
    opt = optax.adam(3e-4)
    opt_state = opt.init(params)
    norm = None

    @jax.jit
    def train_step(params, opt_state, pix, x, y, yb):
        def loss_fn(p):
            act, bi = model.apply(p, pix, x)
            la, lb = jnp.mean((act - y) ** 2), jnp.mean((bi - yb) ** 2)
            return la + a.w_aux * lb, (la, lb)
        (_, (la, lb)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(params, upd), opt_state, la, lb

    t0 = time.time()
    for it in range(a.iters):
        beta = 1.0 if it == 0 else 0.0  # iteration 0 = teacher rollouts (behavior cloning), then student-driven
        student = make_student_act(params, norm, a.enc) if it > 0 else None
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
            turn += np.where(alive, np.nan_to_num(np.asarray(s.metrics["turn"])), 0.0)  # ended episodes may go NaN
            err.append(np.where(alive, np.asarray(s.metrics["track_err"]), np.nan))
            alive &= np.asarray(s.done) == 0
        roll = (f"iter {it} ({'teacher' if beta else 'student'} rollouts): pair turn {turn.mean():.2f} rad/ep, "
                f"drop {1 - alive.mean():.2f}, track_err {1000 * np.nanmean(err):.1f} mm, data {n:,}")
        if norm is None:
            norm = (jnp.array(prop[:n].mean(0)), jnp.array(prop[:n].std(0) + 1e-3))
        rng = np.random.default_rng(it)
        for g in range(a.grad_steps):
            i = rng.integers(0, n, a.batch)
            ii = [i]
            for _ in range(a.stack - 1):
                ii.append(prev[ii[-1]])
            pix = jnp.array(np.concatenate([frames[j] for j in ii], -1))
            params, opt_state, la, lb = train_step(params, opt_state, pix, (jnp.array(prop[i]) - norm[0]) / norm[1],
                                                   jnp.array(label[i]), jnp.array(ball[i]))
        line = f"{time.time() - t0:6.0f}s {roll}; train action mse {float(la):.4f}, ball mse {float(lb):.4f}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        for name in (f"student_{it}.pkl", "student.pkl"):
            with open(os.path.join(a.out, name), "wb") as f:
                pickle.dump({"params": params, "norm": norm, "enc": a.enc, "task": task_kw(a)}, f)
