"""Exploit checks for a pen policy (CPU MuJoCo): yaw trace, COM path in the palm frame, angular velocity split into
palm-normal vs pen-axis components, and the instantaneous rotation centre. Example: python -m rl.pen_check --params ..."""
import argparse
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from rl import pen_env as E
from sim import Sim

p = argparse.ArgumentParser()
p.add_argument("--params", default="out/rl/pen/best_params.pkl")
p.add_argument("--dir", type=float, default=1.0)
p.add_argument("--seconds", type=float, default=20.0)
a = p.parse_args()

net = ppo_networks.make_ppo_networks(73, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128))
policy = jax.jit(ppo_networks.make_inference_fn(net)(pickle.load(open(a.params, "rb")), deterministic=True))
obs_fn = jax.jit(E.observe)
m = E.build_model()
lo, hi = m.actuator_ctrlrange.T
s = Sim(m)
s.reset(hand_q=E.DEFAULT_POSE)
Rh = s.d.body("hand_base").xmat.reshape(3, 3).copy()   # palm frame: x = palm normal, z = fingers, y = thumb side
oh = s.d.body("hand_base").xpos.copy()
n = Rh[:, 0]                                            # palm normal in world (= +z)
palm = s.site_pos("palm")
dt, steps = 0.005, int(a.seconds / 0.005)
yaw_prev = float(E.yaw(E.pen_axis(s.d.qpos[23:27])))
yaw_sum, wz_int, rows, key = 0.0, 0.0, [], jax.random.PRNGKey(0)
for t in range(steps):
    if t % 10 == 0:
        act, _ = policy(obs_fn(s.d.qpos, s.d.qvel, s.d.ctrl, a.dir), key)
        s.d.ctrl[:] = np.clip(s.d.ctrl + 0.15 * np.asarray(act), lo, hi)
    s.step()
    R = s.d.body("pen").xmat.reshape(3, 3)
    ax = R[:, 2]
    w = R @ s.d.qvel[23:26]                              # free-joint angular velocity is in the body frame
    v = s.d.qvel[20:23]                                  # linear velocity of the pen origin, world frame
    wn, wa = w @ n, w @ ax
    y = float(np.arctan2(ax[1], ax[0]))
    yaw_sum += (y - yaw_prev + np.pi) % (2 * np.pi) - np.pi
    yaw_prev = y
    wz_int += wn * dt
    pc = s.d.qpos[20:23].copy()
    c = pc + np.cross(n, v) / wn if abs(wn) > 1.0 else np.full(3, np.nan)   # planar instantaneous rotation centre
    tips = {f: np.linalg.norm((c - s.site_pos(f + "_tip"))[:2]) for f in ("thumb", "index", "middle", "ring", "pinky")}
    rows.append((t * dt, yaw_sum, wn, wa, (pc - oh) @ Rh, (c - pc) @ ax, (c - oh) @ Rh, np.linalg.norm((c - palm)[:2]),
                 min(tips, key=tips.get), min(tips.values()), abs(ax @ n)))
    if bool(E.dropped_fn(s.d.qpos)):
        print(f"dropped at {t * dt:.2f} s")
        break
s.close()

T = np.array([r[0] for r in rows]); Y = np.array([r[1] for r in rows])
WN = np.array([r[2] for r in rows]); WA = np.array([r[3] for r in rows])
COM = np.array([r[4] for r in rows]); ALONG = np.array([r[5] for r in rows]); CH = np.array([r[6] for r in rows])
DPALM = np.array([r[7] for r in rows]); NEAR = [r[8] for r in rows]; DTIP = np.array([r[9] for r in rows])
TILT = np.array([r[10] for r in rows])
ok = ~np.isnan(ALONG)
print(f"(c) metric check: summed wrapped yaw {np.degrees(Y[-1]):.0f} deg vs integrated palm-normal omega "
      f"{np.degrees(wz_int):.0f} deg over {T[-1]:.1f} s")
print("yaw trace (deg, every 2 s):", [int(np.degrees(Y[int(k / dt)])) for k in np.arange(0, T[-1], 2.0)])
print(f"(a) |omega| split: palm-normal mean |w_n| {np.abs(WN).mean():.2f} rad/s (signed mean {WN.mean():+.2f}), "
      f"pen-axis roll mean |w_a| {np.abs(WA).mean():.2f} rad/s; pen tilt |axis.n| mean {TILT.mean():.2f} max {TILT.max():.2f}")
print("COM in palm frame (x=height above palm origin, y=thumb side, z=toward fingertips), mm:")
print("   mean", (1000 * COM.mean(0)).round(1), " std", (1000 * COM.std(0)).round(1),
      " z range", (1000 * COM[:, 2].min()).round(0), "to", (1000 * COM[:, 2].max()).round(0))
print("   path samples (y, z) every 1 s:", [tuple((1000 * COM[int(k / dt), 1:]).round(0).astype(int)) for k in np.arange(0, T[-1], 1.0)])
print(f"(b) rotation centre, {ok.mean():.2f} of steps with |w_n|>1:")
print("   along the pen from its centre, mm: median |d| %.0f, pct 10/90 of d: %s" % (
      1000 * np.median(np.abs(ALONG[ok])), (1000 * np.percentile(ALONG[ok], [10, 90])).round(0)))
print("   in palm frame (y, z) mm: median", (1000 * np.median(CH[ok][:, 1:], 0)).round(0),
      " palm site (y, z):", (1000 * ((palm - oh) @ Rh)[1:]).round(0))
print("   horizontal distance to palm site: median %.0f mm; to nearest fingertip: median %.0f mm" % (
      1000 * np.median(DPALM[ok]), 1000 * np.median(DTIP[ok])))
names, counts = np.unique(np.array(NEAR)[ok], return_counts=True)
print("   nearest fingertip to the centre:", dict(zip(names, (counts / ok.sum()).round(2))))
