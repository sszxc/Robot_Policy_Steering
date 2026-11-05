"""Baoding with actor-side noise and a pace-capped reward. Dict obs: 'state' (noisy teacher actor), 'vis_state'
(hand proprio + target phase, no ball info), 'privileged_state' (critic), and with render=True 'pixels/view_0'.
v2 options: periods=(lo, hi) samples a per-episode speed command (appended to all obs); warp=True uses MJX-Warp
physics + mesh rendering (needs nworld = the vmapped batch size); cam='wrist' (warp only) swaps the top-down
camera for an oblique wrist camera. tilt_max (rad) samples the palm tilt per episode (gravity direction in the hand
frame, applied as per-body forces); g_obs appends that direction (3) before the command in all obs.
dr=True: per-episode physics randomization (DR ranges below; model fields overridden inside the vmapped step),
0/1/2-step action delay, and a 7-dim DR summary in privileged_state (before g / command)."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from robotics.policies.rl import baoding_env as E, baoding_render as BR
from robotics.policies.scripted import baoding as B
from robotics.sim import scene

BALL_X = 0.043  # ball center height above the hand origin at rest on the palm
CON_PER_WORLD = 64
# wrist camera: aimed at the orbit center from the wrist side, optical axis 55 deg off the palm normal,
# 15 cm away (~13 cm above the palm); fovy 42 deg = square center crop of the D435 color stream
WRIST_ALPHA, WRIST_DIST, WRIST_FOVY = np.radians(55), 0.15, 42.0


def wrist_cam(alpha=WRIST_ALPHA, dist=WRIST_DIST):
    """Wrist camera position and axes (x right, y up, z back) in the hand frame."""
    pos = E.CENTER + dist * np.array([np.cos(alpha), 0, -np.sin(alpha)])
    fwd = (E.CENTER - pos) / dist
    up = np.array([0, 0, 1.0]) - fwd * fwd[2]
    up /= np.linalg.norm(up)
    return pos, np.stack([np.cross(fwd, up), up, -fwd], 1)


# physics DR (per episode): scale ranges unless noted; log-uniform for damping/armature
DR = dict(ball_mass=(0.7, 1.3), hand_fric=(0.5, 1.2), ball_fric=(0.5, 1.5), kp=(0.7, 1.3), damp=(0.5, 2.0),
          arm=(0.5, 2.0), floss=(0.0, 0.01), link_mass=(0.8, 1.2), solref=(0.01, 0.03), delay_p=(0.4, 0.4, 0.2))
TH0 = -E.TILT[0]  # nominal palm tilt: 17 deg, fingers downhill


def gravity_dir(th, ph):
    """Unit gravity in the hand frame for palm tilt th (from palm-up) toward azimuth ph (0 = fingers downhill,
    pi/2 = thumb side downhill, pi = wrist downhill)."""
    return jnp.stack([-jnp.cos(th), jnp.sin(th) * jnp.sin(ph), jnp.sin(th) * jnp.cos(ph)], -1)


def build_model_cam(sim_dt, cam="top"):
    """E.build_model plus one camera: 'top' (matches the ray caster view, fingers up, thumb right) or 'wrist'."""
    m0, R, o = E.build_model(sim_dt)
    spec = B.ball_spec(*E.TILT, solref=(0.02, 1))
    scene.add_hand_self_pairs(spec)
    q = np.zeros(4)
    if cam == "top":
        mujoco.mju_mat2Quat(q, np.stack([R[:, 1], R[:, 2], R[:, 0]], 1).ravel())  # cam x=right, y=up, z=back
        spec.worldbody.add_camera(name="top", pos=o + R @ BR.LOOK + BR.DIST * R[:, 0], quat=q, fovy=np.degrees(BR.FOV))
    else:
        pos, Rc = wrist_cam()
        mujoco.mju_mat2Quat(q, (R @ Rc).ravel())
        spec.worldbody.add_camera(name="wrist", pos=o + R @ pos, quat=q, fovy=WRIST_FOVY)
    m = spec.compile()
    m.geom_contype[:], m.geom_conaffinity[:] = m0.geom_contype, m0.geom_conaffinity
    m.opt.timestep, m.opt.iterations, m.opt.ls_iterations, m.opt.cone = (m0.opt.timestep, m0.opt.iterations,
                                                                         m0.opt.ls_iterations, m0.opt.cone)
    return m, R, o


class BaodingNoisyEnv(Env):
    def __init__(self, period=16.0, w_turn=50.0, render=False, ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15,
                 ball_noise=0.002, ball_bias=0.001, q_noise=0.005, qd_noise=0.1, act_noise=0.05, delay_prob=0.5,
                 kick_prob=0.01, kick_std=0.1, periods=None, warp=False, nworld=None, w_over=0.0, cam="top",
                 tilt_max=None, g_obs=None, g_noise=0.03, dr=False):
        self.warp, self.cmd = warp, periods is not None
        self.mj_model, R, o = build_model_cam(sim_dt, cam) if warp else E.build_model(sim_dt)
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
        self.tilt_max = tilt_max
        self.g_obs = (tilt_max is not None) if g_obs is None else g_obs
        self.noise["g"] = g_noise
        self.g_world = jnp.array(self.mj_model.opt.gravity)
        self.dr = dr
        if dr:
            m = self.mj_model
            names = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
            col = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
            self.ball_geoms = np.array([g for g in range(m.ngeom) if names[g].startswith("ball")])
            self.hand_geoms = np.array([g for g in range(m.ngeom) if col[g] and not names[g].startswith(("ball", "world"))])
            self.ball_bodies = np.array([m.body("ball0").id, m.body("ball1").id])
            self.link_bodies = np.array([b for b in range(m.nbody) if m.body(b).name.startswith("finger")])
            self.m0 = {k: jnp.array(getattr(m, k)) for k in ("geom_friction", "geom_solref", "body_mass", "body_inertia",
                       "actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_armature", "dof_frictionloss")}
        self.scene = BR.make_scene(self.mj_model, R, o) if render and not warp else None
        if render and warp:  # keep self.rc alive: its device buffers are freed with it
            self.rc = mjx.create_render_context(mjm=self.mj_model, nworld=nworld, cam_res=(BR.RES, BR.RES),
                                                use_textures=True, use_shadows=False, render_rgb=True,
                                                render_depth=False, enabled_geom_groups=[0, 2])
            self.rcp = self.rc.pytree()

    @property
    def observation_size(self):
        c = int(self.cmd) + 3 * int(self.g_obs)
        size = {"state": 84 + c, "vis_state": 66 + c, "privileged_state": 85 + c + 7 * int(self.dr)}
        if self.render:
            size["pixels/view_0"] = (BR.RES, BR.RES, 6)
        return size

    @property
    def action_size(self):
        return 20

    @property
    def backend(self):
        return "mjx"

    def _img(self, data):
        if not self.warp:
            return BR.render(data.geom_xpos, data.geom_xmat, self.scene)
        d = mjx.refit_bvh(self.mjx_model, data, self.rcp)
        rgb, _ = mjx.render(self.mjx_model, d, self.rcp)
        return jnp.round(255 * mjx.get_rgb(self.rcp, 0, rgb)).astype(jnp.uint8)

    def _init(self, rng):
        """Random start: hand pose noise, balls on the orbit at a random phase; targets start aligned."""
        r1, r2, r3, r4, r5, r6, r7, r8 = jax.random.split(rng, 8)
        hand_q = jnp.clip(E.DEFAULT_POSE + 0.05 * jax.random.normal(r1, (20,)), self.lo, self.hi)
        phi = jax.random.uniform(r2, (), maxval=2 * jnp.pi)
        yz = E.targets(phi) + 0.002 * jax.random.normal(r3, (2, 2))
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
        g_meas = g + self.noise["g"] * jax.random.normal(k3, (3,))  # per-episode FK / mount calibration error
        info = {"rng": r5, "g": g, "g_meas": g_meas / jnp.linalg.norm(g_meas), "theta": phi, "pair": E.pair_angle(b), "turn": jnp.zeros(()),
                "omega": omega if self.cmd else jnp.asarray(self.omega, jnp.float32),
                "bias": self.noise["bias"] * jax.random.normal(r4, (2, 3)),
                "delay": (jax.random.uniform(jax.random.fold_in(r4, 1)) < self.noise["delay"]).astype(jnp.float32),
                "abuf": jnp.zeros((2, 20)), "hist": jnp.zeros((2, 2, 3)), "frame": jnp.zeros((BR.RES, BR.RES, 3), jnp.uint8)}
        if self.dr:
            info.update(self._sample_dr(r8))
        return qpos, info

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
        k1, k2, k3 = jax.random.split(rng, 3)
        b = E.balls_hand(data.qpos, self.R, self.o)
        meas = b + info["bias"] + self.noise["ball"] * jax.random.normal(k1, (2, 3))
        hist = jnp.where(first, jnp.stack([meas, meas]), info["hist"])
        q = data.qpos[:20] + self.noise["q"] * jax.random.normal(k2, (20,))
        qd = data.qvel[:20] + self.noise["qd"] * jax.random.normal(k3, (20,))
        th = info["theta"]
        goal = jnp.concatenate([10 * (E.targets(th) - E.CENTER[1:]).ravel(), jnp.array([jnp.sin(th), jnp.cos(th)])])
        cmd = (info["omega"] / jnp.pi)[None] if self.cmd else jnp.zeros(0)  # T=2 s -> 1.0
        if self.g_obs:
            cmd, cmd_a = jnp.concatenate([info["g"], cmd]), jnp.concatenate([info["g_meas"], cmd])
        else:
            cmd_a = cmd
        prop = jnp.concatenate([q, 0.1 * qd, data.ctrl])
        balls = 10 * (jnp.concatenate([meas[None], hist]) - E.CENTER).ravel()
        if self.dr:  # DR summary for the critic (~unit scale)
            d = info["dr"]
            drv = jnp.concatenate([d["ball_mass"] - 1, jnp.stack([d["hand_fric"] - 1, d["ball_fric"] - 1, d["kp"].mean() - 1,
                                   jnp.log(d["damp"]).mean(), 50 * (d["solref"] - 0.02)])])
            cmd = jnp.concatenate([drv, cmd])
        priv = jnp.concatenate([E.observe(data.qpos, data.qvel, data.ctrl, th, self.R, self.o),
                                info["delay"][None], 1000 * info["bias"].ravel(), cmd])
        obs = {"state": jnp.concatenate([prop, balls, goal, cmd_a]), "vis_state": jnp.concatenate([prop, goal, cmd_a]),
               "privileged_state": priv}
        info = {**info, "hist": jnp.stack([meas, hist[0]])}
        if self.render:
            img = self._img(data)
            prev = jnp.where(first, img, info["frame"])
            obs["pixels/view_0"] = jnp.concatenate([img, prev], -1)
            info = {**info, "frame": img}
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
        n = self.noise
        act = action + n["act"] * jax.random.normal(k_act, (20,))
        ab = info["abuf"]
        used = jnp.where(info["delay"] > 1.5, ab[1], jnp.where(info["delay"] > 0.5, ab[0], act))
        ctrl = jnp.clip(data.ctrl + self.action_scale * used, self.lo, self.hi)
        # random in-plane kick on one ball
        kk = jax.random.split(k_kick, 3)
        kick = (jax.random.uniform(kk[0]) < n["kick_p"]) * n["kick"] * jax.random.normal(kk[1], (2,))
        v = jnp.concatenate([jnp.zeros(1), kick]) @ self.R.T
        adr = jnp.where(jax.random.uniform(kk[2]) < 0.5, 20, 26)
        qvel = data.qvel + jnp.zeros_like(data.qvel).at[adr + jnp.arange(3)].set(v)
        data = data.replace(ctrl=ctrl, qvel=qvel)
        mdl = self._model(info["dr"]) if self.dr else self.mjx_model
        if self.tilt_max is not None:  # rotated gravity = per-body force at the CoM (hand base is welded)
            f = mdl.body_mass[:, None] * (9.81 * self.R @ info["g"] - self.g_world)
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
                "abuf": jnp.stack([act, ab[0]])}
        obs, info = self._obs(data, info, k_obs, fresh)
        metrics = {**state.metrics, "track_err": err.mean(), "turn": dpair, "dropped": dropped.astype(jnp.float32),
                   "pace": dpair / cap}
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = (dropped | nan).astype(jnp.float32)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward), done=done,
                             metrics=metrics, info=info)
