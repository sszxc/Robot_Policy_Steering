"""Batched MJX eval of noisy-Baoding policies (teacher / vision PPO / DAgger student) + optional video of episode 0
(MuJoCo view | 64 px policy view). Pixels are frozen at reset for the 'frozen' ablation.
--cmd (v2 speed-command policies, MuJoCo Warp): speed grid, mid-episode command switch, sweep video.
Example: python -m robotics.policies.rl.baoding_noisy_eval --teacher ... --vision_rl ... --student ... --video"""
from platforms.paths import output_path
import argparse
import subprocess

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from robotics.policies.rl import baoding_env as E
from robotics.policies.rl.baoding_dagger import load_student
from robotics.policies.rl.baoding_noisy_env import BaodingNoisyEnv
from robotics.policies.rl.baoding_noisy_train import load_policy

T = 400


def set_cmd(s, period):
    """Overrides the speed command (info + last obs entry) with per-env periods (n,)."""
    w = 2 * jnp.pi / jnp.asarray(period, jnp.float32)
    obs = {k: (v.at[:, -1].set(w / jnp.pi) if k != "pixels/view_0" else v) for k, v in s.obs.items()}
    return s.replace(obs=obs, info={**s.info, "omega": w})


def rollout(env, act, n, frozen=False, seed=0, schedule=None):
    """schedule: per-step command period (T,) for --cmd envs, else the env's fixed period."""
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    s = reset(jax.random.split(jax.random.PRNGKey(seed), n))
    pix0 = s.obs.get("pixels/view_0")
    alive, turn, err, qpos, views = np.ones(n, bool), np.zeros((T, n)), [], [], []
    for t in range(T):
        if schedule is not None:
            s = set_cmd(s, np.full(n, schedule[t]))
        obs = {**s.obs, "pixels/view_0": pix0} if frozen else s.obs
        if pix0 is not None:
            views.append(np.asarray(s.obs["pixels/view_0"][0, ..., :3]))
        s = step(s, act(obs))
        turn[t] = np.asarray(s.metrics["turn"]) * alive
        err.append(np.where(alive, np.asarray(s.metrics["track_err"]), np.nan))
        qpos.append(np.asarray(s.pipeline_state.qpos[0]))
        alive &= np.asarray(s.done) == 0
    return turn, 1 - alive, 1000 * np.nanmean(err), qpos, views


def video(env, qpos, views, path, fps=20, labels=None):
    m = env.mj_model
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 480)
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = np.asarray(env.o + env.R @ E.CENTER), 0.32, 180, -60
    w = 960 if views else 480
    up = np.arange(480) * views[0].shape[0] // 480 if views else None   # nearest-neighbor upscale to 480
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x480",
                           "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        img = r.render()
        if views:
            img = np.concatenate([img, views[t][up][:, up]], 1)
        if labels is not None:
            im = Image.fromarray(img)
            ImageDraw.Draw(im).text((12, 10), labels[t], fill=(255, 255, 255), font_size=24)
            img = np.asarray(im)
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    r.close()


def pace(turn, schedule, t0=0, t1=T):
    """Achieved / commanded pair rotation over steps [t0, t1), mean over episodes (drops count as no progress)."""
    return turn[t0:t1].sum(0).mean() / (2 * np.pi / np.asarray(schedule[t0:t1]) * 0.05).sum()


def eval_cmd(a, env, pols):
    grid = [float(x) for x in a.grid]
    switches = [(8.0, 3.0), (3.0, 8.0)]
    for name, act, vis in pols:
        for frozen in ((False, True) if vis else (False,)):
            tag = name + ("[frozen]" if frozen else "")
            for P in grid:
                sch = np.full(T, P)
                turn, drop, err, _, _ = rollout(env, act, a.episodes, frozen, schedule=sch)
                print(f"{tag:18s} T={P:4.1f}s  pace {100 * pace(turn, sch):4.0f}%  drop {drop.mean():.2f}  "
                      f"track_err {err:5.1f} mm", flush=True)
            for p0, p1 in switches:
                sch = np.where(np.arange(T) < T // 2, p0, p1)
                turn, drop, err, _, _ = rollout(env, act, a.episodes, frozen, schedule=sch)
                print(f"{tag:18s} switch T {p0:.0f}->{p1:.0f}s at 10 s: pace before {100 * pace(turn, sch, 100, 200):4.0f}%,"
                      f" first 3 s after {100 * pace(turn, sch, 200, 260):4.0f}%, last 7 s {100 * pace(turn, sch, 260):4.0f}%,"
                      f" drop {drop.mean():.2f}, track_err {err:5.1f} mm", flush=True)
        if a.video:  # command sweep 16 -> 8 -> 4 -> 2 s per revolution, 5 s each
            sch = np.repeat([16.0, 8.0, 4.0, 2.0], T // 4)
            turn, drop, err, qpos, views = rollout(env, act, a.episodes, schedule=sch)
            path = str(output_path(f"rl/baoding_cmd/{name}_sweep.mp4"))
            video(env, qpos, views, path, labels=[f"command: {p:.0f} s / rev" for p in sch])
            print(f"  video {path} (sweep pace {100 * pace(turn, sch):.0f}%)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher")
    p.add_argument("--vision_rl")
    p.add_argument("--student")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--clean", action="store_true", help="all noise off")
    p.add_argument("--video", action="store_true", help="write out/rl/baoding_{noisy,cmd}/<name>*.mp4 (episode 0)")
    p.add_argument("--cmd", action="store_true", help="v2 speed-command policies (MuJoCo Warp env)")
    p.add_argument("--grid", nargs="+", default=[2, 3, 4, 8, 16], help="--cmd: periods (s) to evaluate")
    a = p.parse_args()
    off = dict(ball_noise=0, ball_bias=0, q_noise=0, qd_noise=0, act_noise=0, delay_prob=0, kick_prob=0)
    v2 = dict(periods=(2.0, 16.0), warp=True, nworld=a.episodes) if a.cmd else {}
    env = BaodingNoisyEnv(render=True, **(off if a.clean else {}), **v2)
    pols = []
    if a.teacher:
        pols.append(("teacher", load_policy(a.teacher, False, env), False))
    if a.vision_rl:
        pols.append(("vision_rl", load_policy(a.vision_rl, True, env), True))
    if a.student:
        pols.append(("dagger", load_student(a.student), True))
    if a.cmd:
        eval_cmd(a, env, pols)
    for name, act, vis in ([] if a.cmd else pols):
        target = T * env.omega * env.ctrl_dt
        for frozen in ((False, True) if vis else (False,)):
            turn, drop, err, qpos, views = rollout(env, act, a.episodes, frozen)
            turn = turn.sum(0)
            tag = name + ("[frozen pixels]" if frozen else "") + ("[clean]" if a.clean else "")
            print(f"{tag:28s} pair turn {turn.mean():5.2f} rad/ep ({100 * turn.mean() / target:3.0f}% of target "
                  f"{target:.2f}), median {np.median(turn):5.2f}, drop {drop.mean():.2f}, track_err {err:5.1f} mm", flush=True)
            if a.video and not frozen:
                path = str(output_path(f"rl/baoding_noisy/{name}{'_clean' if a.clean else ''}.mp4"))
                video(env, qpos, views, path)
                print("  video", path)
