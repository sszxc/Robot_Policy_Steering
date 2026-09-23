"""Sim: step a model with position targets, interpolated moves, arm IK, and optional viewer/video/log."""
import os
import subprocess

import mujoco
import numpy as np



class Sim:
    def __init__(self, model, viewer=False, video=None, fps=30, size=(640, 480), cam=None, log=None):
        """cam: dict(lookat, distance, azimuth, elevation) for video/viewer; log: .npz path for qpos/ctrl."""
        self.m, self.d = model, mujoco.MjData(model)
        self.arm = model.nu > 20  # arm actuators come first
        self.hand_ctrl = slice(6, 26) if self.arm else slice(0, 20)
        self.arm_qpos = slice(0, 6)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        for k, v in (cam or {}).items():
            setattr(self.cam, k, v)
        self.viewer = self.proc = None
        if viewer:
            from mujoco import viewer as mjv
            self.viewer = mjv.launch_passive(model, self.d)
            self.viewer.cam.lookat[:], self.viewer.cam.distance = self.cam.lookat, self.cam.distance
            self.viewer.cam.azimuth, self.viewer.cam.elevation = self.cam.azimuth, self.cam.elevation
        if video:
            os.makedirs(os.path.dirname(video) or ".", exist_ok=True)
            self.renderer = mujoco.Renderer(model, size[1], size[0])
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
                 "-pix_fmt", "yuv420p", "-vcodec", "libx264", video], stdin=subprocess.PIPE)
        self.frame_every = max(1, round(1 / (fps * model.opt.timestep)))
        self.log_path, self.log = log, {"t": [], "qpos": [], "qvel": [], "ctrl": []}
        self.nstep = 0

    # ---- low level ----
    def reset(self, arm_q=None, hand_q=None):
        mujoco.mj_resetData(self.m, self.d)
        if arm_q is not None:
            self.d.qpos[self.arm_qpos] = arm_q
            self.d.ctrl[:6] = arm_q
        if hand_q is not None:
            self.set_hand_qpos(hand_q)
            self.d.ctrl[self.hand_ctrl] = hand_q
        mujoco.mj_forward(self.m, self.d)

    def set_hand_qpos(self, q):
        for i, jid in enumerate(self.m.actuator_trnid[self.hand_ctrl, 0]):
            self.d.qpos[self.m.jnt_qposadr[jid]] = q[i]

    def hand_q(self):
        return np.array([self.d.qpos[self.m.jnt_qposadr[j]] for j in self.m.actuator_trnid[self.hand_ctrl, 0]])

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.nstep += 1
            if self.nstep % self.frame_every == 0:
                self._frame()

    def _frame(self):
        if self.viewer is not None:
            self.viewer.sync()
        if self.proc is not None:
            self.renderer.update_scene(self.d, self.cam)
            self.proc.stdin.write(self.renderer.render().tobytes())
        if self.log_path:
            for k, v in (("t", self.d.time), ("qpos", self.d.qpos), ("qvel", self.d.qvel), ("ctrl", self.d.ctrl)):
                self.log[k].append(np.copy(v))

    def hold(self, seconds):
        self.step(int(seconds / self.m.opt.timestep))

    # ---- motion ----
    def move(self, arm_q=None, hand_q=None, duration=1.0, callback=None):
        """Smoothly (cosine) interpolate arm/hand position targets from current ctrl to the given ones."""
        c0 = self.d.ctrl.copy()
        c1 = c0.copy()
        if arm_q is not None:
            c1[:6] = arm_q
        if hand_q is not None:
            c1[self.hand_ctrl] = hand_q
        n = max(1, int(duration / self.m.opt.timestep))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)
            self.d.ctrl[:] = c0 + s * (c1 - c0)
            if callback:
                callback(self, s)
            self.step()

    def move_hand(self, q, duration=0.5):
        self.move(hand_q=q, duration=duration)

    def ik(self, site, pos, quat=None, q0=None, iters=200, tol=1e-4):
        """Damped least squares IK on the 6 arm joints for a site pose; returns joint angles (not applied)."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        if q0 is not None:
            d.qpos[:6] = q0
        sid = self.m.site(site).id
        jac = np.zeros((6, self.m.nv))
        err = np.zeros(6)
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            err[:3] = np.asarray(pos) - d.site_xpos[sid]
            if quat is not None:
                cur = np.zeros(4)
                mujoco.mju_mat2Quat(cur, d.site_xmat[sid])
                neg = np.zeros(4)
                mujoco.mju_negQuat(neg, cur)
                dq = np.zeros(4)
                mujoco.mju_mulQuat(dq, np.asarray(quat, float), neg)
                mujoco.mju_quat2Vel(err[3:], dq, 1.0)
            else:
                err[3:] = 0
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.m, d, jac[:3], jac[3:], sid)
            J = jac[:, :6] if quat is not None else jac[:3, :6]
            e = err if quat is not None else err[:3]
            dq6 = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            d.qpos[:6] += np.clip(dq6, -0.2, 0.2)
            d.qpos[:6] = np.clip(d.qpos[:6], *self.m.jnt_range[:6].T)
        self.ik_err = np.linalg.norm(err)
        return d.qpos[:6].copy()

    def move_site(self, site, pos, quat=None, duration=1.0, hand_q=None):
        """IK for a site pose, then interpolate the arm there in joint space."""
        q = self.ik(site, pos, quat, q0=self.d.ctrl[:6])
        self.move(arm_q=q, hand_q=hand_q, duration=duration)
        return q

    # ---- queries ----
    def body_pos(self, name):
        return self.d.body(name).xpos.copy()

    def body_quat(self, name):
        return self.d.body(name).xquat.copy()

    def site_pos(self, name):
        return self.d.site(name).xpos.copy()

    def hand_contacts(self, body):
        """Names of hand bodies currently touching `body`."""
        bid = self.m.body(body).id
        out = set()
        for c in self.d.contact[: self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if bid in (b1, b2):
                name = self.m.body(b2 if b1 == bid else b1).name
                if name.startswith(("finger_r_", "hand_base")):
                    out.add(name)
        return out

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()
        if self.log_path:
            np.savez(self.log_path, **{k: np.array(v) for k, v in self.log.items()})
