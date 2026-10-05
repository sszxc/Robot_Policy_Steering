"""Run a trained pen policy in CPU MuJoCo, record video, report rotation, drops and finger contacts.
Example: MUJOCO_GL=egl python -m robotics.policies.rl.pen_eval --params out/rl/pen/params.pkl"""
from platforms.paths import output_path
import argparse
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small MLP; leaves the GPU to training

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from robotics.policies.rl import pen_env as E
from robotics.sim import Sim

p = argparse.ArgumentParser()
p.add_argument("--params", default=str(output_path('rl/pen/best_params.pkl')))
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--video", default=str(output_path('rl/pen/pen.mp4')))
p.add_argument("--viewer", action="store_true")
p.add_argument("--condim", type=int, default=3)
a = p.parse_args()

net = ppo_networks.make_ppo_networks(73, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128))
policy = jax.jit(ppo_networks.make_inference_fn(net)(pickle.load(open(a.params, "rb")), deterministic=True))
obs_fn = jax.jit(E.observe)

m = E.build_model(condim=a.condim)
lo, hi = m.actuator_ctrlrange.T
cam = dict(lookat=E.PEN_START + [0.01, 0, 0.01], distance=0.34, azimuth=180, elevation=-55)
FINGERS = {"thumb": range(1, 5), "index": range(5, 9), "middle": range(9, 13), "ring": range(13, 17),
           "pinky": range(17, 21)}
rng, out = np.random.default_rng(0), []
for ep in range(a.episodes):
    d = 1.0 if ep % 2 == 0 else -1.0
    s = Sim(m, viewer=a.viewer, video=a.video.replace(".mp4", f"_{'ccw' if d > 0 else 'cw'}.mp4") if ep < 2 else None, cam=cam)
    s.reset(hand_q=np.clip(E.DEFAULT_POSE, lo, hi))
    if ep >= 2:  # same 5 mm xy pen noise as the MJX env reset
        s.d.qpos[20:22] += rng.normal(0, 0.005, 2)
    yaw, turn, dropped, touch, key = float(E.yaw(E.pen_axis(s.d.qpos[23:27]))), 0.0, False, [], jax.random.PRNGKey(ep)
    for t in range(400):
        act, _ = policy(obs_fn(s.d.qpos, s.d.qvel, s.d.ctrl, d), key)
        s.d.ctrl[:] = np.clip(s.d.ctrl + 0.15 * np.asarray(act), lo, hi)
        s.step(10)
        new = float(E.yaw(E.pen_axis(s.d.qpos[23:27])))
        turn += d * ((new - yaw + np.pi) % (2 * np.pi) - np.pi)
        yaw = new
        links = s.hand_contacts("pen")
        touch.append([any(f"finger_r_link{i}" in links for i in r) for r in FINGERS.values()] + ["hand_base" in links])
        if bool(E.dropped_fn(s.d.qpos)):
            dropped = True
            s.hold(0.5)
            break
    s.close()
    dur = (t + 1) * 0.05
    frac = np.mean(touch, 0)
    out.append((turn, dur, dropped))
    print(f"episode {ep} dir {d:+.0f}: rotation {np.degrees(turn):6.0f} deg in {dur:4.1f} s ({turn / dur:5.2f} rad/s, "
          f"{turn / (2 * np.pi):4.1f} rev), dropped {dropped}, contact frac "
          + " ".join(f"{k[0]}{v:.2f}" for k, v in zip(list(FINGERS) + ["palm"], frac)))
t = np.array(out)
print(f"mean {t[:, 0].mean() / (2 * np.pi):.2f} rev/ep, {np.mean(t[:, 0] / t[:, 1]):.2f} rad/s, drop rate {t[:, 2].mean():.2f}")
