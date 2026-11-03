"""Baoding with actor-side noise and a pace-capped reward. Dict obs: 'state' (noisy teacher actor), 'vis_state'
(hand proprio + target phase, no ball info), 'privileged_state' (critic), and with render=True 'pixels/view_0'.
v2 options: periods=(lo, hi) samples a per-episode speed command (appended to all obs); warp=True uses MJX-Warp
physics + mesh rendering (needs nworld = the vmapped batch size)."""
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


def build_model_cam(sim_dt):
    """E.build_model plus a top-down 'top' camera matching the ray caster view (fingers up, thumb right)."""
    m0, R, o = E.build_model(sim_dt)
    spec = B.ball_spec(*E.TILT, solref=(0.02, 1))
    scene.add_hand_self_pairs(spec)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([R[:, 1], R[:, 2], R[:, 0]], 1).ravel())  # cam x=right, y=up, z=back
    spec.worldbody.add_camera(name="top", pos=o + R @ BR.LOOK + BR.DIST * R[:, 0], quat=q, fovy=np.degrees(BR.FOV))
    m = spec.compile()
    m.geom_contype[:], m.geom_conaffinity[:] = m0.geom_contype, m0.geom_conaffinity
    m.opt.timestep, m.opt.iterations, m.opt.ls_iterations, m.opt.cone = (m0.opt.timestep, m0.opt.iterations,
                                                                         m0.opt.ls_iterations, m0.opt.cone)
    return m, R, o


class BaodingNoisyEnv(Env):
    def __init__(self, period=16.0, w_turn=50.0, render=False, ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15,
                 ball_noise=0.002, ball_bias=0.001, q_noise=0.005, qd_noise=0.1, act_noise=0.05, delay_prob=0.5,
                 kick_prob=0.01, kick_std=0.1, periods=None, warp=False, nworld=None, w_over=0.0):
        self.warp, self.cmd = warp, periods is not None
        self.mj_model, R, o = build_model_cam(sim_dt) if warp else E.build_model(sim_dt)
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
        self.scene = BR.make_scene(self.mj_model, R, o) if render and not warp else None
        if render and warp:  # keep self.rc alive: its device buffers are freed with it
            self.rc = mjx.create_render_context(mjm=self.mj_model, nworld=nworld, cam_res=(BR.RES, BR.RES),
                                                use_textures=True, use_shadows=False, render_rgb=True,
                                                render_depth=False, enabled_geom_groups=[0, 2])
            self.rcp = self.rc.pytree()

    @property
    def observation_size(self):
        c = int(self.cmd)
        size = {"state": 84 + c, "vis_state": 66 + c, "privileged_state": 85 + c}
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
        r1, r2, r3, r4, r5, r6 = jax.random.split(rng, 6)
        hand_q = jnp.clip(E.DEFAULT_POSE + 0.05 * jax.random.normal(r1, (20,)), self.lo, self.hi)
        phi = jax.random.uniform(r2, (), maxval=2 * jnp.pi)
        yz = E.targets(phi) + 0.002 * jax.random.normal(r3, (2, 2))
        p = jnp.concatenate([jnp.full((2, 1), BALL_X), yz], 1) @ self.R.T + self.o
        qpos = jnp.concatenate([hand_q, self.q_ball.at[0:3].set(p[0]).at[7:10].set(p[1])])
        b = E.balls_hand(qpos, self.R, self.o)
        # speed command: period log-uniform in `periods` (v1: fixed period)
        omega = 2 * jnp.pi / jnp.exp(jax.random.uniform(r6, (), minval=self.log_T[0], maxval=self.log_T[1]))
        info = {"rng": r5, "theta": phi, "pair": E.pair_angle(b), "turn": jnp.zeros(()),
                "omega": omega if self.cmd else jnp.asarray(self.omega, jnp.float32),
                "bias": self.noise["bias"] * jax.random.normal(r4, (2, 3)),
                "delay": (jax.random.uniform(jax.random.fold_in(r4, 1)) < self.noise["delay"]).astype(jnp.float32),
                "prev_act": jnp.zeros(20), "hist": jnp.zeros((2, 2, 3)), "frame": jnp.zeros((BR.RES, BR.RES, 3), jnp.uint8)}
        return qpos, info

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
        prop = jnp.concatenate([q, 0.1 * qd, data.ctrl])
        balls = 10 * (jnp.concatenate([meas[None], hist]) - E.CENTER).ravel()
        priv = jnp.concatenate([E.observe(data.qpos, data.qvel, data.ctrl, th, self.R, self.o),
                                info["delay"][None], 1000 * info["bias"].ravel(), cmd])
        obs = {"state": jnp.concatenate([prop, balls, goal, cmd]), "vis_state": jnp.concatenate([prop, goal, cmd]),
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
        used = jnp.where(info["delay"] > 0, info["prev_act"], act)
        ctrl = jnp.clip(data.ctrl + self.action_scale * used, self.lo, self.hi)
        # random in-plane kick on one ball
        kk = jax.random.split(k_kick, 3)
        kick = (jax.random.uniform(kk[0]) < n["kick_p"]) * n["kick"] * jax.random.normal(kk[1], (2,))
        v = jnp.concatenate([jnp.zeros(1), kick]) @ self.R.T
        adr = jnp.where(jax.random.uniform(kk[2]) < 0.5, 20, 26)
        qvel = data.qvel + jnp.zeros_like(data.qvel).at[adr + jnp.arange(3)].set(v)
        data = data.replace(ctrl=ctrl, qvel=qvel)
        data = jax.lax.fori_loop(0, self.n_frames, lambda _, d: mjx.step(self.mjx_model, d), data)

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

        info = {**info, "rng": rng, "theta": theta, "pair": pair, "turn": info["turn"] + dpair, "prev_act": act}
        obs, info = self._obs(data, info, k_obs, fresh)
        metrics = {**state.metrics, "track_err": err.mean(), "turn": dpair, "dropped": dropped.astype(jnp.float32),
                   "pace": dpair / cap}
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = (dropped | nan).astype(jnp.float32)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward), done=done,
                             metrics=metrics, info=info)
