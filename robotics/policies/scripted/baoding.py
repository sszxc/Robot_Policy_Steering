"""Baoding balls: two balls orbit each other in the (tilted, palm-up) hand via a periodic finger wave.

Gait = sinusoidal traveling wave over index..pinky plus a thumb sweep; parameters found by CEM (--search).
Run: python -m robotics.policies.scripted.baoding [--video] [--viewer] [--search]
"""
from platforms.paths import output_path
import sys

import mujoco
import numpy as np

from robotics.sim import Sim, cli_sim, hand as H, scene

R, MASS = 0.02, 0.06          # ball radius (m) / mass (kg)
DURATION = 20.0               # evaluation length (s)
# Params: a pitch (fingers-down +), b roll (thumb-side-down +), f Hz; per finger k: mcp/pip offset+amp, phase;
# thumb (planar 3R in the palm plane): cmc/pitch/ip offset+amp+phase, fixed yaw.
FINGER_LO, FINGER_HI = [0.2, 0.0, 0.0, 0.0, -3.14], [1.8, 0.8, 1.5, 0.8, 3.14]
THUMB_LO = [-1.3, 0.0, -3.14, 0.0, 0.0, -3.14, 0.0, 0.0, -3.14, -0.6]
THUMB_HI = [1.0, 1.0, 3.14, 1.57, 0.8, 3.14, 1.57, 0.8, 3.14, 0.0]
LOW = np.array([-0.5, -0.4, 0.3] + FINGER_LO * 4 + THUMB_LO)
HIGH = np.array([0.2, 0.4, 1.5] + FINGER_HI * 4 + THUMB_HI)
# ball start centers in the hand frame (x palm normal, y thumb side, z fingers)
BALL_START = [np.array([0.021 + R + 0.002, 0.024, 0.075]), np.array([0.021 + R + 0.002, -0.024, 0.075])]
# best CEM result (sustained only ~145 deg / 20 s; scripted baseline, RL does the real task: rl/baoding_*)
PARAMS = [-0.389, -0.173, 0.357, 0.907, 0.632, 0.402, 0.168, 2.413, 1.383, 0.269, 0.41, 0.258, 0.996, 0.848, 0.227, 0.447, 0.174, -0.885, 1.486, 0.704, 0.554, 0.605, -2.017, -0.267, 0.114, -1.535, 0.956, 0.271, -0.063, 1.029, 0.257, 1.745, -0.388]


def tilt_quat(a, b):
    """Palm-up hand pitched by a about the thumb axis, then rolled by b about the finger axis."""
    def rot(v, k, th):
        return v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * (k @ v) * (1 - np.cos(th))
    t = np.array([0, -1.0, 0])
    n, f = rot(np.array([0, 0, 1.0]), t, a), rot(np.array([1.0, 0, 0]), t, a)
    return scene.hand_quat(rot(n, f, b), f)


def build(a, b, solref=(0.005, 1)):
    return ball_spec(a, b, solref).compile()


def ball_spec(a, b, solref=(0.005, 1)):
    """Tilted hand-only scene with two balls side by side near the finger roots."""
    q = tilt_quat(a, b)
    spec = scene.hand_only(pos=(0, 0, 0.3), quat=q)
    Rh = np.zeros(9)
    mujoco.mju_quat2Mat(Rh, q)
    Rh = Rh.reshape(3, 3)
    for i, rgba in enumerate([(0.85, 0.2, 0.2, 1), (0.2, 0.4, 0.9, 1)]):
        p = np.array([0, 0, 0.3]) + Rh @ BALL_START[i]
        body = spec.worldbody.add_body(name=f"ball{i}", pos=p.tolist())
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[R], mass=MASS, rgba=list(rgba),
                      condim=4, friction=[0.6, 0.005, 0.0001], solref=list(solref))
    return spec


def gait(p, t):
    w = 2 * np.pi * p[2] * t
    fingers = []
    for k in range(4):
        m0, m1, p0, p1, ph = p[3 + 5 * k: 8 + 5 * k]
        pip = p0 + p1 * np.sin(w + ph + 0.5 * np.pi)
        fingers.append((0, m0 + m1 * np.sin(w + ph), pip, 0.7 * pip))
    c0, c1, cph, q0, q1, qph, i0, i1, iph, yaw = p[23:33]
    thumb = (c0 + c1 * np.sin(w + cph), yaw, q0 + q1 * np.sin(w + qph), i0 + i1 * np.sin(w + iph))
    lo, hi = HAND_RANGE
    return np.clip(H.pose(thumb, *fingers), lo, hi)


_m = build(0, 0)
HAND_RANGE = _m.actuator_ctrlrange[:, 0], _m.actuator_ctrlrange[:, 1]


def ball_plane(s):
    """Ball centers in the hand frame (y, z) = palm-plane coordinates, plus height above palm (x)."""
    hb = s.d.body("hand_base")
    Rm, o = hb.xmat.reshape(3, 3), hb.xpos
    return [Rm.T @ (s.d.body(f"ball{i}").xpos - o) for i in range(2)]


def rollout(p, sim=None, duration=DURATION, trace=None):
    """Run the gait; returns (cumulative pair-axis rotation rad, dropped?, number of half-swaps).
    A ball leaving the palm region or riding on the other ball (stacked) counts as dropped."""
    s = sim or Sim(build(p[0], p[1]))
    s.reset(hand_q=gait(p, 0))
    s.hold(0.5)
    t0 = s.d.time
    ang, prev = 0.0, None
    n = int(duration / s.m.opt.timestep)
    for i in range(n):
        s.d.ctrl[s.hand_ctrl] = gait(p, s.d.time - t0)
        s.step()
        if i % 10:
            continue
        b0, b1 = ball_plane(s)
        if any(b[0] < 0 or b[0] > 0.021 + R + 0.012 or abs(b[1]) > 0.09 or not -0.02 < b[2] < 0.2 for b in (b0, b1)):
            return ang, True, int(abs(ang) // np.pi)
        a = np.arctan2(b1[2] - b0[2], b1[1] - b0[1])
        if prev is not None:
            ang += (a - prev + np.pi) % (2 * np.pi) - np.pi
        prev = a
        if trace is not None:
            trace.append(ang)
    return ang, False, int(abs(ang) // np.pi)


def _score(p, duration=12.0, windows=4):
    """Sustained rotation: the weakest of `windows` equal time windows (same direction), minus a drop penalty."""
    tr = []
    ang, dropped, _ = rollout(p, duration=duration, trace=tr)
    if dropped:
        return -6.0 + len(tr) / 600
    ix = np.linspace(0, len(tr) - 1, windows + 1).astype(int)
    return min(np.diff(np.array(tr)[ix]) * np.sign(ang))


def search(iters=80, pop=256, elite=24, seed=0):
    """Cross-entropy search over gait parameters (CPU, multiprocessing)."""
    from multiprocessing import Pool
    rng = np.random.default_rng(seed)
    mu, sd = (LOW + HIGH) / 2, (HIGH - LOW) / 3
    best, best_s = None, -np.inf
    output_path("scripted/baoding").mkdir(parents=True, exist_ok=True)
    with Pool(16) as pool:
        for it in range(iters):
            X = np.clip(mu + sd * rng.standard_normal((pop, len(mu))), LOW, HIGH)
            S = np.array(pool.map(_score, X))
            idx = np.argsort(S)[-elite:]
            mu, sd = X[idx].mean(0), 0.7 * sd + 0.3 * X[idx].std(0) + 1e-3
            if S[idx[-1]] > best_s:
                best, best_s = X[idx[-1]].copy(), S[idx[-1]]
            print(f"iter {it}: best {S.max():.2f} mean elite {S[idx].mean():.2f} overall {best_s:.2f}", flush=True)
            np.save(str(output_path('scripted/baoding/best.npy')), best)
    print("PARAMS =", np.round(best, 3).tolist())
    return best


def main():
    if "--search" in sys.argv:
        search()
        return
    p = np.array(PARAMS)
    m = build(p[0], p[1])
    cam = dict(lookat=np.array([0.05, 0.0, 0.33]), distance=0.38, azimuth=200, elevation=-50)
    s = cli_sim(m, "baoding", cam)
    ang, dropped, swaps = rollout(p, sim=s)
    s.close()
    print(f"cumulative pair rotation {np.degrees(ang):.0f} deg in {DURATION:.0f} s, half-swaps {swaps}, dropped {dropped}")
    print("SUCCESS" if swaps >= 2 and not dropped else "FAIL")


if __name__ == "__main__":
    main()
