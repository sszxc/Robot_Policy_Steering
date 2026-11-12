"""Real wrist-camera data for the Baoding sim (ros2_ur out/calib): undistort to the sim pinhole, background library.
Run: python -m robotics.policies.rl.baoding_real --bg   -> out/rl/baoding_real/bg.npy (N, 128, 176, 3) uint8 RGB
     python -m robotics.policies.rl.baoding_real --viz  -> docs/results/baoding_real_vis.html (real vs sim images)
     python -m robotics.policies.rl.baoding_real --check STUDENT.pkl -> open-loop ball positions on the real ball frames"""
import argparse
import base64
import glob
import json

import cv2
import numpy as np

from platforms.paths import output_path
from robotics.sim import scene

CALIB = "/home/lab/Documents/ros2_ur/out/calib"
BG = output_path("rl/baoding_real/bg.npy")


def undistort(img, w=88, h=64):
    """Real 640x480 BGR frame -> the sim 'wrist' pinhole (f = w * 32 / 88 px) at w x h, area-averaged."""
    c = json.load(open(scene.MJ / "lab" / "wrist_camera.json"))
    f = 4 * w * 32 / 88  # remap at 4x, then average down
    Kn = np.array([[f, 0, (4 * w - 1) / 2], [0, f, (4 * h - 1) / 2], [0, 0, 1]])
    m1, m2 = cv2.initUndistortRectifyMap(np.array(c["K"]), np.array(c["dist"]), None, Kn, (4 * w, 4 * h), cv2.CV_32FC1)
    return cv2.resize(cv2.remap(img, m1, m2, cv2.INTER_LINEAR), (w, h), interpolation=cv2.INTER_AREA)


def viz(calib, out, n=24):
    """HTML page: real frames, the sim render before / after the background composite, the backgrounds alone."""
    import jax
    from robotics.policies.rl.baoding_noisy_env import BaodingNoisyEnv
    env = BaodingNoisyEnv(render=True, periods=(2.0, 16.0), warp=True, nworld=n, cam="wrist", tilt_max=np.radians(35),
                          tilt_about="nominal", vis_dr=True)
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    shot = lambda: np.asarray(jax.jit(jax.vmap(env.reset))(keys).obs["pixels/view_0"])[..., :3]
    raw = shot()  # no library yet: rays that miss keep the flat key color
    lib = np.load(BG)
    env.bg = jax.numpy.array(lib)
    comp = shot()
    bgs = np.asarray(jax.vmap(env._bg)(jax.random.split(jax.random.PRNGKey(1), n)))
    bgs = (255 * bgs.reshape(n, 64, 2, 88, 2, 3).mean((2, 4))).astype(np.uint8)
    real = np.stack([undistort(cv2.imread(f))[..., ::-1] for f in sorted(glob.glob(f"{calib}/balls/*.png"))])
    lib = lib[:: len(lib) // 8][:8, ::2, ::2]

    def stat(x):  # mean RGB of white-ball-like and yellow-ball-like pixels
        x = x.reshape(-1, 3).astype(float)
        w, y = x[x.min(1) > 150], x[(x[:, 0] > 150) & (x[:, 1] > 120) & (x[:, 2] < 110)]
        return f"<td>{w.mean(0).round().astype(int)}</td><td>{y.mean(0).round().astype(int)}</td>"

    def row(title, x):
        g = np.vstack([np.hstack(list(x[r:r + 8])) for r in range(0, len(x), 8)])
        g = cv2.resize(g[..., ::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        b = base64.b64encode(cv2.imencode(".png", g)[1]).decode()
        return f'<h3>{title}</h3><img src="data:image/png;base64,{b}" style="max-width:100%">'

    html = ["<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:1450px;margin:auto'>",
            "<h2>Baoding wrist camera: real frames vs sim visual randomization (policy image 88x64)</h2>",
            row("1. Real frames (ros2_ur out/calib/balls), undistorted to the sim pinhole", real[[3, 4, 9, 12, 16, 20, 26, 30]]),
            row("2. Sim, visual DR without the background (missed rays = flat key color)", raw[:8]),
            row("3. Same episodes after the background composite", comp[:8]),
            row("4. More composited episodes", comp[8:]),
            row("5. Background library: real lab frames (8 of %d)" % len(np.load(BG)), lib),
            row("6. Per-episode backgrounds alone: 70% real frame (rotation, zoom, shift, flip, gain), 30% smooth colors", bgs),
            "<h3>7. Ball pixel colors (mean sRGB)</h3><table border=1 cellpadding=4><tr><th></th><th>white-like</th>"
            "<th>yellow-like</th></tr>", f"<tr><td>real</td>{stat(real)}</tr><tr><td>sim composite</td>{stat(comp)}</tr></table>",
            "</body></html>"]
    open(out, "w").write("\n".join(html))
    print(out)


def check(calib, student, out):
    """Open loop on the real labelled ball frames (white = ball0, yellow = ball1): the student's ball-position head
    vs the position from the clicked circle (center ray + apparent radius of the 20 mm ball)."""
    import pickle
    import jax.numpy as jnp
    from robotics.policies.rl import baoding_env as E
    from robotics.policies.rl.baoding_dagger import Student
    from robotics.policies.rl.baoding_noisy_env import TH0, gravity_dir
    from robotics.policies.scripted import baoding as B
    c = json.load(open(scene.MJ / "lab" / "wrist_camera.json"))
    K, dist, T = np.array(c["K"]), np.array(c["dist"]), np.array(c["T_hand_cam"])
    d = pickle.load(open(student, "rb"))
    nf = d["params"]["params"]["Conv_0"]["kernel"].shape[2] // 3 if d.get("enc", "flat") == "flat" else None
    frames = {f["file"]: f for f in json.load(open(f"{calib}/balls/frames.json"))["frames"]}
    goal = np.concatenate([10 * (np.asarray(E.targets(0.0)) - E.CENTER[1:]).ravel(), [0.0, 1.0]])
    ray = lambda uv: (lambda v: v / np.linalg.norm(v))(np.append(cv2.undistortPoints(np.float64([[uv]]), K, dist).ravel(), 1))
    tiles, errs = [], []
    for name, cs in json.load(open(f"{calib}/balls/circles.json")).items():
        img = undistort(cv2.imread(f"{calib}/balls/{name}"))
        q = np.array(frames[name]["hand"])
        prop = np.concatenate([q, np.zeros(20), q, goal, np.asarray(gravity_dir(TH0, 0.0)), [0.25]])
        x = (jnp.array(prop, jnp.float32) - d["norm"][0]) / d["norm"][1]
        for n in ([nf] if nf else (2, 3)):  # ssm weights do not fix the stack: take the one that fits the MLP
            try:
                pred = Student(d.get("enc", "flat")).apply(d["params"], jnp.array(np.tile(img[..., ::-1], n))[None], x[None])[1]
                break
            except Exception:
                continue
        pred = np.asarray(pred).reshape(2, 3) / 10 + E.CENTER
        big = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        for i, (lab, col) in enumerate((("white", (255, 0, 0)), ("yellow", (0, 0, 255)))):
            pc = T[:3, :3].T @ (pred[i] - T[:3, 3])
            u, v = 32 * pc[0] / pc[2] + 43.5, 32 * pc[1] / pc[2] + 31.5
            cv2.drawMarker(big, (int(4 * u + 2), int(4 * v + 2)), col, cv2.MARKER_CROSS, 14, 2)
            if lab in cs:
                cx, cy, r = cs[lab]
                r0, r1 = ray((cx, cy)), ray((cx + r, cy))
                true = T[:3, :3] @ (B.R / np.sin(np.arccos(r0 @ r1)) * r0) + T[:3, 3]
                errs.append((lab, 1000 * (pred[i] - true)))
        tiles.append(big)
    for lab in ("white", "yellow"):
        e = np.array([v for k, v in errs if k == lab])
        print(f"{lab}: n {len(e)}, |error| median {np.median(np.linalg.norm(e, axis=1)):.1f} mm, "
              f"mean error (palm normal, thumb, fingers) {e.mean(0).round(1)} mm, std {e.std(0).round(1)}")
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]))
    print(out, "(cross = predicted ball center: blue white ball, red yellow ball)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bg", action="store_true", help="build the background library from <calib>/background/*.png")
    p.add_argument("--viz", action="store_true", help="real vs sim image page (needs --bg first)")
    p.add_argument("--check", help="student .pkl: open-loop ball-position check on <calib>/balls")
    p.add_argument("--calib", default=CALIB)
    a = p.parse_args()
    if a.bg:
        files = sorted(glob.glob(f"{a.calib}/background/*.png"))
        bg = np.stack([undistort(cv2.imread(f), 176, 128)[..., ::-1] for f in files])
        BG.parent.mkdir(parents=True, exist_ok=True)
        np.save(BG, bg)
        print(f"{len(files)} frames -> {BG} {bg.shape}")
    if a.viz:
        viz(a.calib, str(scene.MJ.parents[1] / "docs" / "results" / "baoding_real_vis.html"))
    if a.check:
        check(a.calib, a.check, a.check.replace(".pkl", "_real_check.png"))
