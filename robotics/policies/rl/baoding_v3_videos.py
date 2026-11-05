"""Report videos for Baoding v3 (wrist cam / tilt / physics DR / visual DR): panel grids of MuJoCo views with the hand
re-oriented so gravity points down, 64 px policy view inset, running pace / drop label.
Usage: MUJOCO_GL=egl python -m robotics.policies.rl.baoding_v3_videos --only tilt   (one section per process:
MJX-Warp cannot hold two different models in one process; sections: tilt dr vis nominal)"""
from platforms.paths import output_path
import argparse
import os
import subprocess

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from robotics.policies.rl import baoding_env as E
from robotics.policies.rl.baoding_dagger import load_student
from robotics.policies.rl.baoding_noisy_env import BaodingNoisyEnv, gravity_dir
from robotics.policies.rl.baoding_noisy_eval import set_cmd, set_tilt
from robotics.policies.rl.baoding_noisy_train import load_policy

T, PANEL = 400, 360
CK = dict(A="out/rl/baoding_wrist/dagger/student.pkl", B="out/rl/baoding_tilt/dagger/student.pkl",
          C="out/rl/baoding_dr/dagger/student.pkl", D="out/rl/baoding_vis/dagger/student.pkl",
          v2="out/rl/baoding_cmd/teacher/best_params.pkl")
SIDE = {0: "fingers", 90: "thumb", 180: "wrist", 270: "pinky"}


def make_env(n, g_obs=True, dr=False, vis_dr=False):
    return BaodingNoisyEnv(render=True, periods=(2.0, 16.0), warp=True, nworld=n, cam="wrist", tilt_max=np.radians(35),
                           tilt_about="nominal", g_obs=g_obs, dr=dr, vis_dr=vis_dr)


def run(env, act, seed, tilts=None, period=None):
    """n parallel episodes; tilts: list of (deg, azimuth deg) per env (None = sampled), period: s/rev (None = sampled)."""
    n = env.nworld
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    s = reset(jax.random.split(jax.random.PRNGKey(seed), n))
    if tilts is not None:
        s = set_tilt(env, s, jnp.stack([gravity_dir(np.radians(t), np.radians(p)) for t, p in tilts]))
    if period is not None:
        s = set_cmd(s, np.full(n, period))
    qpos, views, cum = [], [], np.zeros((T, n))
    alive, drop_t = np.ones(n, bool), np.full(n, T)
    last_q = np.asarray(s.pipeline_state.qpos)
    for t in range(T):
        views.append(np.asarray(s.obs["pixels/view_0"][..., :3]))
        s = step(s, act(s.obs))
        q = np.asarray(s.pipeline_state.qpos)
        last_q = np.where(alive[:, None], q, last_q)  # freeze the scene once an episode ends
        qpos.append(last_q)
        dturn = np.where(alive, np.nan_to_num(np.asarray(s.metrics["turn"])), 0.0)
        cum[t] = (cum[t - 1] if t else 0) + dturn
        ended = alive & (np.asarray(s.done) > 0)
        drop_t[ended] = t
        alive &= ~ended
    omega = np.asarray(s.info["omega"])
    return dict(qpos=np.stack(qpos), views=np.stack(views), g=np.asarray(s.info["g"]), omega=omega, cum=cum,
                drop_t=drop_t, dr=jax.tree.map(np.asarray, s.info.get("dr", {})), delay=np.asarray(s.info["delay"]))


class View:
    """MuJoCo render of the env model with all non-world geoms rotated about the hand origin so that gravity is down."""
    def __init__(self, env, size=PANEL):
        self.m, self.R, self.o = env.mj_model, np.asarray(env.R), np.asarray(env.o)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, size, size)
        self.body = self.m.geom_bodyid

    def __call__(self, q, g):
        dvec = self.R @ g
        e = np.array([0, 0, -1.0])
        ax, c = np.cross(dvec, e), float(dvec @ e)
        K = np.cross(np.eye(3), ax / (np.linalg.norm(ax) + 1e-12))
        th = np.arctan2(np.linalg.norm(ax), c)
        Q = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.o + Q @ (self.R @ E.CENTER)
        cam.distance, cam.azimuth, cam.elevation = 0.34, 0, -62  # from the wrist side
        self.r.update_scene(self.d, cam)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False  # shadows would not follow the rotated hand
        sc = self.r.scene
        for i in range(sc.ngeom):
            gg = sc.geoms[i]
            if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM and self.body[gg.objid] != 0:
                gg.pos[:] = self.o + Q @ (gg.pos - self.o)
                gg.mat[:] = Q @ gg.mat
        return self.r.render()


def panel(img, inset, title, sub, dropped):
    im = Image.fromarray(img)
    ins = Image.fromarray(inset).resize((128, 128), Image.NEAREST)
    im.paste(ins, (PANEL - 136, PANEL - 136))
    dr = ImageDraw.Draw(im)
    dr.rectangle((0, 0, PANEL, 50), fill=(20, 24, 30))
    dr.text((8, 4), title, fill=(255, 255, 255), font_size=17)
    dr.text((8, 27), sub, fill=(255, 120, 110) if dropped else (170, 230, 190), font_size=15)
    return np.asarray(im)


def tilt_name(t, p):
    return "flat palm" if t == 0 else f"{t:.0f}° {SIDE[p]} down"


def write(path, rows, fps=20):
    """rows: list of lists of per-panel frame functions f(t) -> (PANEL, PANEL, 3)."""
    H, W = PANEL * len(rows), PANEL * max(len(r) for r in rows)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for t in range(T):
        frame = np.zeros((H, W, 3), np.uint8)
        for i, row in enumerate(rows):
            for j, f in enumerate(row):
                frame[i * PANEL:(i + 1) * PANEL, j * PANEL:(j + 1) * PANEL] = f(t)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB", flush=True)


def panels(res, view, names, who):
    """Frame functions for every episode of a run result; label = names[i] + running pace / drop."""
    fs = []
    for i in range(res["qpos"].shape[1]):
        def f(t, i=i):
            pace = res["cum"][t, i] / (res["omega"][i] * 0.05 * (t + 1))
            dropped = res["drop_t"][i] <= t
            sub = f"T={2 * np.pi / res['omega'][i]:.1f} s/rev  pace {100 * pace:3.0f}%" + ("  DROPPED" if dropped else "")
            return panel(view(res["qpos"][t, i], res["g"][i]), res["views"][min(t, res["drop_t"][i]), i],
                         f"{who}: {names[i]}", sub, dropped)
        fs.append(f)
    return fs


def summary(res, name):
    pace = res["cum"][-1] / (res["omega"] * 0.05 * T)
    print(f"  {name}: pace {np.round(100 * pace).astype(int).tolist()} drop_t {res['drop_t'].tolist()}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=["tilt", "dr", "vis"])
    p.add_argument("--out", default=str(output_path("rl/baoding_v3/videos")))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if "tilt" in a.only:  # B0 (v2 teacher, no tilt input) vs B student on the same five tilts, T = 6 s
        tilts = [(30, 0), (0, 0), (30, 90), (15, 270), (15, 180)]
        names = [tilt_name(*x) for x in tilts]
        env = make_env(5)  # one Warp model per process: the v2 teacher gets the state without the gravity entries
        shapes = type("S", (), {"observation_size": {**env.observation_size, "state": env.observation_size["state"] - 3}})
        v2 = load_policy(CK["v2"], False, shapes)
        r0 = run(env, lambda o: v2({**o, "state": jnp.concatenate([o["state"][:, :-4], o["state"][:, -1:]], 1)}),
                 0, tilts, 6.0)
        summary(r0, "v2 teacher")
        v = View(env)
        r1 = run(env, load_student(CK["B"]), 0, tilts, 6.0)
        summary(r1, "B student")
        write(f"{a.out}/tilt_B0_vs_B.mp4", [panels(r0, v, names, "v2 (no tilt input)"), panels(r1, v, names, "B student")])

    if "dr" in a.only:  # same randomized physics + tilt episodes: B student (never saw DR) vs C student
        env = make_env(4, dr=True)
        v = View(env)
        rb = run(env, load_student(CK["B"]), 7)
        rc = run(env, load_student(CK["C"]), 7)
        names = [f"kp×{rb['dr']['kp'][i].mean():.2f} fric×{rb['dr']['hand_fric'][i]:.2f} delay {rb['delay'][i]:.0f}"
                 for i in range(4)]
        summary(rb, "B student, DR")
        summary(rc, "C student, DR")
        write(f"{a.out}/dr_B_vs_C.mp4", [panels(rb, v, names, "B"), panels(rc, v, names, "C")])

    if "vis" in a.only:  # same visual-DR episodes: B student vs D student (inset = randomized policy view)
        env = make_env(4, dr=True, vis_dr=True)
        v = View(env)
        rb = run(env, load_student(CK["B"]), 11)
        rd = run(env, load_student(CK["D"]), 11)
        names = [f"random ep {i}" for i in range(4)]
        summary(rb, "B student, vis DR")
        summary(rd, "D student, vis DR")
        write(f"{a.out}/vis_B_vs_D.mp4", [panels(rb, v, names, "B"), panels(rd, v, names, "D")])

    if "nominal" in a.only:  # D student on random tilt + DR + visual DR episodes (showcase)
        env = make_env(6, dr=True, vis_dr=True)
        v = View(env)
        rd = run(env, load_student(CK["D"]), 23)
        th = np.degrees(np.arccos(-rd["g"][:, 0]))
        names = [f"tilt {x:.0f}°" for x in th]
        summary(rd, "D student, all randomization")
        fs = panels(rd, v, names, "D")
        write(f"{a.out}/D_showcase.mp4", [fs[:3], fs[3:]])
