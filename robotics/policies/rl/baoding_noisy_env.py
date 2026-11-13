"""Baoding with actor-side noise and a pace-capped reward. Dict obs: 'state' (noisy teacher actor), 'vis_state'
(hand proprio + target phase, no ball info), 'privileged_state' (critic), and with render=True 'pixels/view_0'.
v2 options: periods=(lo, hi) samples a per-episode speed command (appended to all obs); warp=True uses MJX-Warp
physics + mesh rendering (needs nworld = the vmapped batch size); cam='wrist' (warp only) swaps the top-down
camera for the calibrated real wrist camera (88 x 64 px; top is 64 x 64). tilt_max (rad) samples the palm tilt per episode (gravity direction in the hand
frame, applied as per-body forces); g_obs appends that direction (3) before the command in all obs.
dr=True: per-episode physics randomization (DR ranges below; model fields overridden inside the vmapped step),
0/1/2-step action delay, and a 7-dim DR summary in privileged_state (before g / command).
vis_dr=True (warp render): per-episode camera-mount jitter, ball/hand/floor colors, light direction, image
brightness / contrast / white balance / pixel noise; wrist camera also: background composite (bg = real-frame
library from baoding_real --bg, mixed with smooth random colors) and a 1-2 step image delay.
stack = number of stacked frames in 'pixels/view_0' (newest first).
pert = overrides of PERT (extra perturbations, all off by default; used for sensitivity evals)."""
import jax
import jax.numpy as jnp
import mujoco
from jax.scipy.ndimage import map_coordinates
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from robotics.policies.rl import baoding_env as E, baoding_render as BR
from robotics.policies.scripted import baoding as B
from robotics.sim import scene

BALL_X = 0.043  # ball center height above the hand origin at rest on the palm
CON_PER_WORLD = 64


def wrist_cam():
    """The calibrated real wrist camera (lab/ur5.xml, camera 'wrist' on hand_mount): position and axes (x right,
    y up, z back) in the hand frame, fovy (deg) and resolution (w, h)."""
    arm = mujoco.MjSpec.from_file(str(scene.MJ / "lab" / "ur5.xml"))
    c, Rc = arm.camera("wrist"), np.zeros(9)
    mujoco.mju_quat2Mat(Rc, c.quat)
    return c.pos - arm.site("hand_attach").pos, Rc.reshape(3, 3), c.fovy, tuple(int(v) for v in c.resolution)


# physics DR (per episode): scale ranges unless noted; log-uniform for damping/armature
DR = dict(ball_mass=(0.9, 1.1), hand_fric=(0.5, 1.2), ball_fric=(0.5, 1.5), kp=(0.7, 1.3), damp=(0.5, 2.0),
          arm=(0.5, 2.0), floss=(0.0, 0.01), link_mass=(0.8, 1.2), solref=(0.01, 0.03), delay_p=(0.4, 0.4, 0.2))
VIS = dict(cam_pos=0.005, cam_rot=np.radians(3), ball_gain=(0.75, 1.15), ball_tint=(0.93, 1.07), hand_gray=(0.03, 0.25),
           floor=(0.3, 1.2), light=np.radians(40), bright=0.1, contrast=(0.8, 1.2), wb=(0.93, 1.07), noise=0.03,
           bg_real=0.7, bg_rot=np.radians(40), bg_zoom=(1.0, 1.6), bg_shift=0.25, bg_gain=(0.6, 1.3),
           img_delay=(0.7, 0.3))  # image age at the policy: 1 or 2 control steps (real: 35-70 ms + inference)
# extra perturbations (off by default). Forces in ball weights (m g) / N, angles in rad.
PERT = dict(
    f_ball=0.0,      # white 3D force on each ball, new every control step (std, ball weights)
    push=0.0,        # sustained 3D push on one ball (std, ball weights); starts with prob push_p, lasts push_n steps
    f_link=0.0,      # sustained 3D push on one finger link (std, N), same timing as push
    spin=0.0,        # angular-velocity kick on one ball at each push start (std, rad/s)
    push_p=0.02, push_n=6,
    g_jit=0.0,       # in-episode wander of the gravity direction (std, ~1 s correlation); needs tilt
    q_bias=0.0,      # per-episode joint zero offset (std): shifts the observed q and ctrl
    prop_delay=0,    # 1: observed q / qd are one control step old
    delay_jit=0.0,   # per-step prob. that the action delay is re-drawn from 0/1/2 for that step
    init_ball=0.002, init_phase=0.0,  # start: ball position noise (std, m), target phase offset (std)
    backlash=0.0,    # total play between commanded and effective position target
    rate=None,       # joint target rate limit (rad/s)
    # model-level (same for all episodes)
    tors=1.0,        # torsional friction scale (all geoms)
    roll=None,       # rolling friction coefficient (m): contacts become condim 6
    ball_r=1.0, ball_I=1.0, frc=1.0,  # ball radius / ball inertia / actuator force limit scales
    palm_fric=1.0, finger_fric=1.0)   # sliding friction scales by hand part
SS = 2  # wrist camera: render at SS x the policy resolution, composite the background, average down
TH0 = -E.TILT[0]  # nominal palm tilt: 17 deg, fingers downhill


def gravity_dir(th, ph):
    """Unit gravity in the hand frame for palm tilt th (from palm-up) toward azimuth ph (0 = fingers downhill,
    pi/2 = thumb side downhill, pi = wrist downhill)."""
    return jnp.stack([-jnp.cos(th), jnp.sin(th) * jnp.sin(ph), jnp.sin(th) * jnp.cos(ph)], -1)


def build_model_cam(sim_dt, cam="top"):
    """E.build_model plus one camera: 'top' (matches the ray caster view, fingers up, thumb right) or 'wrist'.
    -> model, R, o, image (w, h)."""
    m0, R, o = E.build_model(sim_dt)
    spec = B.ball_spec(*E.TILT, solref=(0.02, 1))
    scene.add_hand_self_pairs(spec)
    q, res = np.zeros(4), (BR.RES, BR.RES)
    if cam == "top":
        mujoco.mju_mat2Quat(q, np.stack([R[:, 1], R[:, 2], R[:, 0]], 1).ravel())  # cam x=right, y=up, z=back
        spec.worldbody.add_camera(name="top", pos=o + R @ BR.LOOK + BR.DIST * R[:, 0], quat=q, fovy=np.degrees(BR.FOV))
    else:
        pos, Rc, fovy, res = wrist_cam()
        spec.geom("floor").group = 1  # not rendered: rays that miss (depth 0) are the background mask
        mujoco.mju_mat2Quat(q, (R @ Rc).ravel())
        spec.worldbody.add_camera(name="wrist", pos=o + R @ pos, quat=q, fovy=fovy)
        # fill light along the view direction: the lab's diffuse light leaves no dark side on the real balls
        spec.worldbody.add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=-(R @ Rc)[:, 2])
    m = spec.compile()
    m.geom_contype[:], m.geom_conaffinity[:] = m0.geom_contype, m0.geom_conaffinity
    m.opt.timestep, m.opt.iterations, m.opt.ls_iterations, m.opt.cone = (m0.opt.timestep, m0.opt.iterations,
                                                                         m0.opt.ls_iterations, m0.opt.cone)
    return m, R, o, res


class BaodingNoisyEnv(Env):
    def __init__(self, period=16.0, w_turn=50.0, render=False, ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15,
                 ball_noise=0.002, ball_bias=0.001, q_noise=0.005, qd_noise=0.1, act_noise=0.05, delay_prob=0.5,
                 kick_prob=0.01, kick_std=0.1, periods=None, warp=False, nworld=None, w_over=0.0, cam="top",
                 tilt_max=None, g_obs=None, g_noise=0.03, dr=False, vis_dr=False,
                 tilt_about="up", bg=None, stack=2, pert=None):
        self.warp, self.cmd = warp, periods is not None
        self.mj_model, R, o, self.res = (build_model_cam(sim_dt, cam) if warp
                                         else (*E.build_model(sim_dt), (BR.RES, BR.RES)))  # res = image (w, h)
        assert not set(pert or {}) - set(PERT), set(pert) - set(PERT)
        self.pert = P = {**PERT, **(pert or {})}
        m = self.mj_model
        names = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        col = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        self.ball_geoms = np.array([g for g in range(m.ngeom) if names[g].startswith("ball")])
        self.hand_geoms = np.array([g for g in range(m.ngeom) if col[g] and not names[g].startswith(("ball", "world"))])
        self.ball_bodies = np.array([m.body("ball0").id, m.body("ball1").id])
        self.link_bodies = np.array([b for b in range(m.nbody) if m.body(b).name.startswith("finger")])
        finger = np.array([names[g].startswith("finger") for g in self.hand_geoms])
        m.geom_friction[self.hand_geoms[finger], 0] *= P["finger_fric"]
        m.geom_friction[self.hand_geoms[~finger], 0] *= P["palm_fric"]
        m.geom_friction[:, 1] *= P["tors"]
        if P["roll"] is not None:  # the contact uses the larger coefficient of the two geoms: set all
            m.geom_condim[np.concatenate([self.ball_geoms, self.hand_geoms])] = 6
            m.geom_friction[:, 2] = P["roll"]
        m.geom_size[self.ball_geoms, 0] *= P["ball_r"]
        m.geom_rbound[self.ball_geoms] *= P["ball_r"]
        m.geom_aabb[self.ball_geoms, 3:] *= P["ball_r"]
        m.body_inertia[self.ball_bodies] *= P["ball_I"]
        m.actuator_forcerange[:] *= P["frc"]
        self.xf = any(P[k] > 0 for k in ("f_ball", "push", "f_link"))
        self.R, self.o = jnp.array(R), jnp.array(o)
        self.mjx_model = mjx.put_model(self.mj_model, impl="warp" if warp else "jax")
        self.n_frames = int(round(ctrl_dt / sim_dt))
        self.omega, self.ctrl_dt, self.action_scale, self.w_turn = 2 * np.pi / period, ctrl_dt, action_scale, w_turn
        self.log_T = np.log(periods if self.cmd else (period, period))
        self.lo, self.hi = jnp.array(self.mj_model.actuator_ctrlrange.T)
        self.q_ball = jnp.array(self.mj_model.qpos0[20:])
        self.noise = dict(ball=ball_noise, bias=ball_bias, q=q_noise, qd=qd_noise, act=act_noise, delay=delay_prob,
                          kick_p=kick_prob, kick=kick_std)
        self.render, self.nworld, self.w_over = render, nworld, w_over
        self.tilt_max, self.tilt_about = tilt_max, tilt_about  # cap axis: "up" (palm up) or "nominal" (TH0)
        self.g_obs = (tilt_max is not None) if g_obs is None else g_obs
        self.noise["g"] = g_noise
        self.g_world = jnp.array(self.mj_model.opt.gravity)
        self.dr, self.vis_dr, self.stack = dr, vis_dr, stack
        self.ss = SS if warp and cam == "wrist" else 1
        self.bg = None if bg is None else jnp.array(np.load(bg))  # (N, SS*h, SS*w, 3) uint8
        if vis_dr:
            m = self.mj_model
            self.ball_vgeoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("ball")])
            self.hand_vgeoms = np.array([g for g in range(m.ngeom) if m.geom_group[g] == 2])
            self.rgba0, self.mat_rgba0 = jnp.array(m.geom_rgba), jnp.array(m.mat_rgba)
        if dr:
            self.m0 = {k: jnp.array(getattr(m, k)) for k in ("geom_friction", "geom_solref", "body_mass", "body_inertia",
                       "actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_armature", "dof_frictionloss")}
        self.scene = BR.make_scene(self.mj_model, R, o) if render and not warp else None
        if render and warp:  # keep self.rc alive: its device buffers are freed with it
            self.rc = mjx.create_render_context(mjm=self.mj_model, nworld=nworld,
                                                cam_res=(self.ss * self.res[0], self.ss * self.res[1]),
                                                use_textures=True, use_shadows=False, render_rgb=True,
                                                render_depth=self.ss > 1, enabled_geom_groups=[0, 2])
            self.rcp = self.rc.pytree()

    @property
    def observation_size(self):
        c = int(self.cmd) + 3 * int(self.g_obs)
        size = {"state": 84 + c, "vis_state": 66 + c, "privileged_state": 85 + c + 7 * int(self.dr)}
        if self.render:
            size["pixels/view_0"] = (self.res[1], self.res[0], 3 * self.stack)
        return size

    @property
    def action_size(self):
        return 20

    @property
    def backend(self):
        return "mjx"

    def _img(self, data, info, rng):
        if not self.warp:
            return BR.render(data.geom_xpos, data.geom_xmat, self.scene)
        m = self.mjx_model
        if self.vis_dr:  # camera-mount jitter (pose read by the renderer from data) + per-episode colors
            v = info["vis"]
            rv = v["cam_rot"]
            th = jnp.linalg.norm(rv) + 1e-9
            K = jnp.cross(jnp.eye(3), rv / th)
            Rj = jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * K @ K
            data = data.replace(cam_xpos=data.cam_xpos + self.R @ v["cam_dp"], cam_xmat=data.cam_xmat @ Rj)
            data = data.tree_replace({"_impl.light_xdir": data._impl.light_xdir @ v["light"].T})
            rgba = self.rgba0.at[self.ball_vgeoms, :3].multiply(v["ball_gain"])
            rgba = rgba.at[self.hand_vgeoms, :3].set(jnp.clip(v["hand_rgb"], 0, 1))
            m = m.tree_replace({"geom_rgba": rgba,
                                "mat_rgba": self.mat_rgba0.at[0, :3].multiply(v["floor"])})
        d = mjx.refit_bvh(m, data, self.rcp)
        rgb, depth = mjx.render(m, d, self.rcp)
        x = mjx.get_rgb(self.rcp, 0, rgb)
        if self.ss > 1:
            if self.vis_dr and self.bg is not None:
                x = jnp.where(mjx.get_depth(self.rcp, 0, depth, 1.0) > 0, x, self._bg(v["bg_key"]))
            x = x.reshape(self.res[1], self.ss, self.res[0], self.ss, 3).mean((1, 3))
        if self.vis_dr:  # photometric: contrast about the image mean, brightness, white balance, pixel noise
            x = (x - x.mean()) * v["contrast"] + x.mean() + v["bright"]
            x = x * v["wb"] + v["noise"] * jax.random.normal(rng, x.shape)
        return jnp.round(255 * jnp.clip(x, 0, 1)).astype(jnp.uint8)

    def _bg(self, key):
        """Per-episode background at render resolution: a real lab frame (random rotation / zoom / shift / flip,
        mirrored at the borders) or smooth random colors, times a random gain."""
        h, w = self.ss * self.res[1], self.ss * self.res[0]
        k = jax.random.split(key, 9)
        u = lambda key, lo, hi, shape=(): jax.random.uniform(key, shape, minval=lo, maxval=hi)
        yy, xx = jnp.meshgrid(jnp.arange(h) - (h - 1) / 2, jnp.arange(w) - (w - 1) / 2, indexing="ij")
        rot, zoom = u(k[0], -VIS["bg_rot"], VIS["bg_rot"]), u(k[1], *VIS["bg_zoom"])
        flip = jnp.where(jax.random.uniform(k[2]) < 0.5, -1.0, 1.0)
        dx, dy = u(k[3], -VIS["bg_shift"], VIS["bg_shift"], (2,)) * jnp.array([w, h])
        c, s = jnp.cos(rot), jnp.sin(rot)
        xs, ys = flip * (c * xx - s * yy) / zoom + (w - 1) / 2 + dx, (s * xx + c * yy) / zoom + (h - 1) / 2 + dy
        img = self.bg[jax.random.randint(k[4], (), 0, self.bg.shape[0])].astype(jnp.float32) / 255
        real = jnp.stack([map_coordinates(img[..., i], [ys, xs], 1, mode="mirror") for i in range(3)], -1)
        coarse = jax.image.resize(jax.random.uniform(k[5], (3, 4, 3)), (h, w, 3), "cubic")
        fine = jax.image.resize(jax.random.uniform(k[6], (12, 16, 3)), (h, w, 3), "linear")
        smooth = coarse + u(k[7], 0.0, 0.6) * (fine - 0.5)
        smooth = smooth.mean(-1, keepdims=True) + u(jax.random.fold_in(key, 2), 0.0, 1.0) * (smooth - smooth.mean(-1, keepdims=True))
        return jnp.clip(jnp.where(jax.random.uniform(k[8]) < VIS["bg_real"], real, smooth)
                        * u(jax.random.fold_in(key, 1), *VIS["bg_gain"]), 0, 1)

    def _init(self, rng):
        """Random start: hand pose noise, balls on the orbit at a random phase; targets start aligned."""
        r1, r2, r3, r4, r5, r6, r7, r8 = jax.random.split(rng, 8)
        hand_q = jnp.clip(E.DEFAULT_POSE + 0.05 * jax.random.normal(r1, (20,)), self.lo, self.hi)
        phi = jax.random.uniform(r2, (), maxval=2 * jnp.pi)
        P = self.pert
        yz = E.targets(phi) + P["init_ball"] * jax.random.normal(r3, (2, 2))
        p = jnp.concatenate([jnp.full((2, 1), BALL_X), yz], 1) @ self.R.T + self.o
        qpos = jnp.concatenate([hand_q, self.q_ball.at[0:3].set(p[0]).at[7:10].set(p[1])])
        b = E.balls_hand(qpos, self.R, self.o)
        # speed command: period log-uniform in `periods` (v1: fixed period)
        omega = 2 * jnp.pi / jnp.exp(jax.random.uniform(r6, (), minval=self.log_T[0], maxval=self.log_T[1]))
        # palm tilt: uniform over the spherical cap tilt <= tilt_max (fixed nominal if None)
        k1, k2, k3 = jax.random.split(r7, 3)
        if self.tilt_max is None:
            th, ph = jnp.asarray(TH0), jnp.zeros(())
        else:
            th = jnp.arccos(jax.random.uniform(k1, (), minval=np.cos(self.tilt_max), maxval=1.0))
            ph = jax.random.uniform(k2, (), maxval=2 * jnp.pi)
        g = gravity_dir(th, ph)
        if self.tilt_about == "nominal":  # rotate the cap axis from palm-up to the nominal 17 deg fingers-downhill
            c, s_ = np.cos(TH0), np.sin(TH0)
            g = jnp.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]) @ g
        g_meas = g + self.noise["g"] * jax.random.normal(k3, (3,))  # per-episode FK / mount calibration error
        info = {"rng": r5, "g": g, "g_meas": g_meas / jnp.linalg.norm(g_meas),
                "theta": phi + P["init_phase"] * jax.random.normal(jax.random.fold_in(r2, 1)), "pair": E.pair_angle(b), "turn": jnp.zeros(()),
                "omega": omega if self.cmd else jnp.asarray(self.omega, jnp.float32),
                "bias": self.noise["bias"] * jax.random.normal(r4, (2, 3)),
                "delay": (jax.random.uniform(jax.random.fold_in(r4, 1)) < self.noise["delay"]).astype(jnp.float32),
                "abuf": jnp.zeros((2, 20)), "hist": jnp.zeros((2, 2, 3)),
                "ctrl": hand_q, "prop": jnp.zeros(40), "q_bias": P["q_bias"] * jax.random.normal(jax.random.fold_in(r1, 1), (20,)),
                "push_f": jnp.zeros((self.mj_model.nbody, 3)), "push_t": jnp.zeros(()), "g_ou": jnp.zeros(3),
                "frames": jnp.zeros((self.stack + 1, self.res[1], self.res[0], 3), jnp.uint8)}  # previous images
        if self.dr:
            info.update(self._sample_dr(r8))
        if self.vis_dr:
            info["vis"] = self._sample_vis(jax.random.fold_in(r8, 1))
        return qpos, info

    def _sample_vis(self, rng):
        k = jax.random.split(rng, 16)
        u = lambda key, lo_hi, shape=(): jax.random.uniform(key, shape, minval=lo_hi[0], maxval=lo_hi[1])
        ax = jax.random.normal(k[1], (3,))
        lv = jax.random.normal(k[10], (3,))
        lv = lv / jnp.linalg.norm(lv) * u(k[11], (0.0, VIS["light"]))  # light directions: one random rotation
        th = jnp.linalg.norm(lv) + 1e-9
        K = jnp.cross(jnp.eye(3), lv / th)
        return {"light": jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * K @ K, "bg_key": k[12],
                "img_delay": 1 + jax.random.choice(k[13], 2, p=jnp.array(VIS["img_delay"])),
                "cam_dp": u(k[0], (-VIS["cam_pos"], VIS["cam_pos"]), (3,)),
                "cam_rot": ax / jnp.linalg.norm(ax) * u(k[2], (0.0, VIS["cam_rot"])),
                "ball_gain": u(k[3], VIS["ball_gain"], (2, 1)) * u(k[14], VIS["ball_tint"], (2, 3)),
                "hand_rgb": u(k[4], VIS["hand_gray"]) + u(k[5], (-0.02, 0.02), (3,)),
                "floor": u(k[6], VIS["floor"], (3,)), "bright": u(k[7], (-VIS["bright"], VIS["bright"])),
                "contrast": u(k[8], VIS["contrast"]), "wb": u(k[9], VIS["wb"], (3,)),
                "noise": u(jax.random.fold_in(rng, 1), (0.0, VIS["noise"]))}

    def _sample_dr(self, rng):
        k = jax.random.split(rng, 10)
        u = lambda key, lo_hi, shape=(): jax.random.uniform(key, shape, minval=lo_hi[0], maxval=lo_hi[1])
        lu = lambda key, lo_hi, shape=(): jnp.exp(u(key, np.log(lo_hi), shape))
        dr = {"ball_mass": u(k[0], DR["ball_mass"], (2,)), "hand_fric": u(k[1], DR["hand_fric"]),
              "ball_fric": u(k[2], DR["ball_fric"]), "kp": u(k[3], DR["kp"], (20,)), "damp": lu(k[4], DR["damp"], (20,)),
              "arm": lu(k[5], DR["arm"], (20,)), "floss": u(k[6], DR["floss"], (20,)),
              "link_mass": u(k[7], DR["link_mass"], (len(self.link_bodies),)), "solref": u(k[8], DR["solref"])}
        delay = jax.random.choice(k[9], 3, p=jnp.array(DR["delay_p"])).astype(jnp.float32)
        return {"dr": dr, "delay": delay}

    def _model(self, dr):
        """Per-episode model: nominal fields scaled by the DR sample (inside vmap the overridden fields are batched)."""
        m0 = self.m0
        fr = m0["geom_friction"].at[self.hand_geoms, 0].multiply(dr["hand_fric"])
        fr = fr.at[self.ball_geoms, 0].multiply(dr["ball_fric"])
        sr = m0["geom_solref"].at[self.ball_geoms, 0].set(dr["solref"])
        ms = jnp.ones(m0["body_mass"].shape[0]).at[self.ball_bodies].set(dr["ball_mass"])
        ms = ms.at[self.link_bodies].set(dr["link_mass"])
        g, b = m0["actuator_gainprm"], m0["actuator_biasprm"]
        g = g.at[:, 0].multiply(dr["kp"])
        b = b.at[:, 1].multiply(dr["kp"]).at[:, 2].multiply(jnp.sqrt(dr["kp"]))  # keep the damping ratio
        nh = 20  # hand dofs come first; ball free-joint dofs keep their nominal values
        return self.mjx_model.tree_replace({
            "geom_friction": fr, "geom_solref": sr, "body_mass": m0["body_mass"] * ms,
            "body_inertia": m0["body_inertia"] * ms[:, None], "actuator_gainprm": g, "actuator_biasprm": b,
            "dof_damping": m0["dof_damping"].at[:nh].multiply(dr["damp"]),
            "dof_armature": m0["dof_armature"].at[:nh].multiply(dr["arm"]),
            "dof_frictionloss": m0["dof_frictionloss"].at[:nh].set(dr["floss"])})

    def _obs(self, data, info, rng, first):
        """Builds the obs dict and updates the ball history / previous frame in info."""
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        b = E.balls_hand(data.qpos, self.R, self.o)
        meas = b + info["bias"] + self.noise["ball"] * jax.random.normal(k1, (2, 3))
        hist = jnp.where(first, jnp.stack([meas, meas]), info["hist"])
        cur = jnp.concatenate([data.qpos[:20], data.qvel[:20]])
        old = jnp.where(first, cur, info["prop"]) if self.pert["prop_delay"] else cur
        q = old[:20] + info["q_bias"] + self.noise["q"] * jax.random.normal(k2, (20,))
        qd = old[20:] + self.noise["qd"] * jax.random.normal(k3, (20,))
        th = info["theta"]
        goal = jnp.concatenate([10 * (E.targets(th) - E.CENTER[1:]).ravel(), jnp.array([jnp.sin(th), jnp.cos(th)])])
        cmd = (info["omega"] / jnp.pi)[None] if self.cmd else jnp.zeros(0)  # T=2 s -> 1.0
        if self.g_obs:
            cmd, cmd_a = jnp.concatenate([info["g"], cmd]), jnp.concatenate([info["g_meas"], cmd])
        else:
            cmd_a = cmd
        prop = jnp.concatenate([q, 0.1 * qd, info["ctrl"] + info["q_bias"]])
        balls = 10 * (jnp.concatenate([meas[None], hist]) - E.CENTER).ravel()
        if self.dr:  # DR summary for the critic (~unit scale)
            d = info["dr"]
            drv = jnp.concatenate([d["ball_mass"] - 1, jnp.stack([d["hand_fric"] - 1, d["ball_fric"] - 1, d["kp"].mean() - 1,
                                   jnp.log(d["damp"]).mean(), 50 * (d["solref"] - 0.02)])])
            cmd = jnp.concatenate([drv, cmd])
        priv = jnp.concatenate([E.observe(data.qpos, data.qvel, info["ctrl"], th, self.R, self.o),
                                info["delay"][None], 1000 * info["bias"].ravel(), cmd])
        obs = {"state": jnp.concatenate([prop, balls, goal, cmd_a]), "vis_state": jnp.concatenate([prop, goal, cmd_a]),
               "privileged_state": priv}
        info = {**info, "hist": jnp.stack([meas, hist[0]]), "prop": cur}
        if self.render:
            img = self._img(data, info, k4)
            buf = jnp.concatenate([img[None], jnp.where(first, img, info["frames"])])  # newest first
            d = info["vis"]["img_delay"] if self.vis_dr and self.ss > 1 else 0
            obs["pixels/view_0"] = jnp.concatenate(list(jax.lax.dynamic_slice_in_dim(buf, d, self.stack)), -1)
            info = {**info, "frames": buf[:-1]}
        return obs, info

    def reset(self, rng):
        qpos, info = self._init(rng)
        data = (mjx.make_data(self.mj_model, impl="warp", naconmax=self.nworld * CON_PER_WORLD, njmax=4 * CON_PER_WORLD)
                if self.warp else mjx.make_data(self.mjx_model))
        data = mjx.forward(self.mjx_model, data.replace(qpos=qpos, ctrl=qpos[:20]))
        obs, info = self._obs(data, info, jax.random.fold_in(rng, 7), True)
        metrics = {"track_err": jnp.zeros(()), "turn": jnp.zeros(()), "dropped": jnp.zeros(()),
                   "pace": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        data, info = state.pipeline_state, state.info
        # brax AutoResetWrapper restores a fixed first state and keeps info: re-sample the start ourselves
        # (only qpos/qvel/ctrl: MJX-Warp contact buffers are shared across worlds)
        if "steps" in info:
            fresh = info["steps"] == 0
            q0, i0 = self._init(info["rng"])
            data = data.replace(qpos=jnp.where(fresh, q0, data.qpos), qvel=jnp.where(fresh, 0.0, data.qvel),
                                ctrl=jnp.where(fresh, q0[:20], data.ctrl))
            info = {**info, **jax.tree.map(lambda a, b: jnp.where(fresh, a, b), i0, {k: info[k] for k in i0})}
        else:
            fresh = jnp.array(False)
        rng, k_act, k_kick, k_obs = jax.random.split(info["rng"], 4)
        n, P = self.noise, self.pert
        kp = jax.random.split(jax.random.fold_in(info["rng"], 1), 10)
        act = action + n["act"] * jax.random.normal(k_act, (20,))
        ab = info["abuf"]
        delay = jnp.where(jax.random.uniform(kp[0]) < P["delay_jit"], jax.random.randint(kp[1], (), 0, 3), info["delay"])
        used = jnp.where(delay > 1.5, ab[1], jnp.where(delay > 0.5, ab[0], act))
        dq = self.action_scale * used
        if P["rate"] is not None:
            dq = jnp.clip(dq, -P["rate"] * self.ctrl_dt, P["rate"] * self.ctrl_dt)
        cmd = jnp.clip(info["ctrl"] + dq, self.lo, self.hi)  # commanded target (observed)
        # backlash: the effective target only moves once the command leaves the play around it
        ctrl = jnp.clip(data.ctrl, cmd - P["backlash"] / 2, cmd + P["backlash"] / 2) if P["backlash"] > 0 else cmd
        # random in-plane kick on one ball
        kk = jax.random.split(k_kick, 3)
        kick = (jax.random.uniform(kk[0]) < n["kick_p"]) * n["kick"] * jax.random.normal(kk[1], (2,))
        v = jnp.concatenate([jnp.zeros(1), kick]) @ self.R.T
        adr = jnp.where(jax.random.uniform(kk[2]) < 0.5, 20, 26)
        qvel = data.qvel + jnp.zeros_like(data.qvel).at[adr + jnp.arange(3)].set(v)
        mdl = self._model(info["dr"]) if self.dr else self.mjx_model
        # pushes: at a start, one ball and one finger link get a new force held for push_n steps (+ a ball spin kick)
        start = (jax.random.uniform(kp[2]) < P["push_p"]) & (info["push_t"] <= 0)
        w_ball = 9.81 * mdl.body_mass[self.ball_bodies]
        ib = jax.random.randint(kp[3], (), 0, 2)
        new = jnp.zeros_like(info["push_f"]).at[jnp.array(self.ball_bodies)[ib]].set(
            P["push"] * w_ball[ib] * jax.random.normal(kp[4], (3,)))
        new = new.at[jnp.array(self.link_bodies)[jax.random.randint(kp[5], (), 0, len(self.link_bodies))]].set(
            P["f_link"] * jax.random.normal(kp[6], (3,)))
        push_f, push_t = jnp.where(start, new, info["push_f"]), jnp.where(start, P["push_n"], info["push_t"] - 1)
        qvel = qvel.at[23 + 6 * ib + jnp.arange(3)].add(start * P["spin"] * jax.random.normal(kp[7], (3,)))
        data = data.replace(ctrl=ctrl, qvel=qvel)
        g_ou = 0.95 * info["g_ou"] + 0.31 * P["g_jit"] * jax.random.normal(kp[8], (3,))
        f = (push_t > 0) * push_f
        f = f.at[self.ball_bodies].add(P["f_ball"] * w_ball[:, None] * jax.random.normal(kp[9], (2, 3)))
        if self.tilt_max is not None:  # rotated gravity = per-body force at the CoM (hand base is welded)
            g = info["g"] + g_ou
            f = f + mdl.body_mass[:, None] * (9.81 * self.R @ (g / jnp.linalg.norm(g)) - self.g_world)
        if self.tilt_max is not None or self.xf:
            data = data.replace(xfrc_applied=jnp.concatenate([f, jnp.zeros_like(f)], 1))
        data = jax.lax.fori_loop(0, self.n_frames, lambda _, d: mjx.step(mdl, d), data)

        cap = info["omega"] * self.ctrl_dt
        theta = info["theta"] + cap
        b = E.balls_hand(data.qpos, self.R, self.o)
        err = jnp.linalg.norm(b[:, 1:] - E.targets(theta), axis=-1)
        pair = E.pair_angle(b)
        dpair = (pair - info["pair"] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        dropped = E.dropped_fn(b)
        # tracking + rotation progress capped at the target pace (faster earns nothing; w_over > 0 penalizes it)
        reward = (jnp.exp(-err / 0.015).sum() + self.w_turn * jnp.clip(dpair, -0.3, cap)
                  - self.w_over * self.w_turn * jnp.clip(dpair - cap, 0.0, 0.3)
                  - 0.001 * jnp.sum(action ** 2) - 50.0 * dropped)

        info = {**info, "rng": rng, "theta": theta, "pair": pair, "turn": info["turn"] + dpair,
                "abuf": jnp.stack([act, ab[0]]), "ctrl": cmd, "push_f": push_f, "push_t": push_t, "g_ou": g_ou}
        obs, info = self._obs(data, info, k_obs, fresh)
        metrics = {**state.metrics, "track_err": err.mean(), "turn": dpair, "dropped": dropped.astype(jnp.float32),
                   "pace": dpair / cap}
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = (dropped | nan).astype(jnp.float32)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward), done=done,
                             metrics=metrics, info=info)
