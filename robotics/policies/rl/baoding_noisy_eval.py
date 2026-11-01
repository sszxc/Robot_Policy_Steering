"""Batched MJX eval of noisy-Baoding policies (teacher / vision PPO / DAgger student) + optional video of episode 0
(MuJoCo view | 64 px policy view). Pixels are frozen at reset for the 'frozen' ablation.
Example: python -m robotics.policies.rl.baoding_noisy_eval --teacher ... --vision_rl ... --student ... --video"""
from platforms.paths import output_path
import argparse
import subprocess

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from robotics.policies.rl import baoding_env as E
from robotics.policies.rl.baoding_dagger import load_student
from robotics.policies.rl.baoding_noisy_env import BaodingNoisyEnv
from robotics.policies.rl.baoding_noisy_train import load_policy

T = 400


def rollout(env, act, n, frozen=False, seed=0):
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    s = reset(jax.random.split(jax.random.PRNGKey(seed), n))
    pix0 = s.obs.get("pixels/view_0")
    alive, turn, err, qpos, views = np.ones(n, bool), np.zeros(n), [], [], []
    for t in range(T):
        obs = {**s.obs, "pixels/view_0": pix0} if frozen else s.obs
        if pix0 is not None:
            views.append(np.asarray(s.obs["pixels/view_0"][0, ..., :3]))
        s = step(s, act(obs))
        turn += np.asarray(s.metrics["turn"]) * alive
        err.append(np.where(alive, np.asarray(s.metrics["track_err"]), np.nan))
        qpos.append(np.asarray(s.pipeline_state.qpos[0]))
        alive &= np.asarray(s.done) == 0
    return turn, 1 - alive, 1000 * np.nanmean(err), qpos, views


def video(env, qpos, views, path, fps=20):
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
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    r.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher")
    p.add_argument("--vision_rl")
    p.add_argument("--student")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--clean", action="store_true", help="all noise off")
    p.add_argument("--video", action="store_true", help="write out/rl/baoding_noisy/<name>.mp4 (episode 0)")
    a = p.parse_args()
    off = dict(ball_noise=0, ball_bias=0, q_noise=0, qd_noise=0, act_noise=0, delay_prob=0, kick_prob=0)
    env = BaodingNoisyEnv(render=True, **(off if a.clean else {}))
    target = T * env.omega * env.ctrl_dt
    pols = []
    if a.teacher:
        pols.append(("teacher", load_policy(a.teacher, False, env), False))
    if a.vision_rl:
        pols.append(("vision_rl", load_policy(a.vision_rl, True, env), True))
    if a.student:
        pols.append(("dagger", load_student(a.student), True))
    for name, act, vis in pols:
        for frozen in ((False, True) if vis else (False,)):
            turn, drop, err, qpos, views = rollout(env, act, a.episodes, frozen)
            tag = name + ("[frozen pixels]" if frozen else "") + ("[clean]" if a.clean else "")
            print(f"{tag:28s} pair turn {turn.mean():5.2f} rad/ep ({100 * turn.mean() / target:3.0f}% of target "
                  f"{target:.2f}), median {np.median(turn):5.2f}, drop {drop.mean():.2f}, track_err {err:5.1f} mm", flush=True)
            if a.video and not frozen:
                path = str(output_path(f"rl/baoding_noisy/{name}{'_clean' if a.clean else ''}.mp4"))
                video(env, qpos, views, path)
                print("  video", path)
