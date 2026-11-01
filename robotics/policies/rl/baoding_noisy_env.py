"""Baoding with actor-side noise and a pace-capped reward. Dict obs: 'state' (noisy teacher actor), 'vis_state'
(hand proprio + target phase, no ball info), 'privileged_state' (critic), and with render=True 'pixels/view_0'."""
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from robotics.policies.rl import baoding_env as E, baoding_render as BR

BALL_X = 0.043  # ball center height above the hand origin at rest on the palm


class BaodingNoisyEnv(Env):
    def __init__(self, period=16.0, w_turn=50.0, render=False, ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15,
                 ball_noise=0.002, ball_bias=0.001, q_noise=0.005, qd_noise=0.1, act_noise=0.05, delay_prob=0.5,
                 kick_prob=0.01, kick_std=0.1):
        self.mj_model, R, o = E.build_model(sim_dt)
        self.R, self.o = jnp.array(R), jnp.array(o)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.n_frames = int(round(ctrl_dt / sim_dt))
        self.omega, self.ctrl_dt, self.action_scale, self.w_turn = 2 * np.pi / period, ctrl_dt, action_scale, w_turn
        self.lo, self.hi = jnp.array(self.mj_model.actuator_ctrlrange.T)
        self.q_ball = jnp.array(self.mj_model.qpos0[20:])
        self.noise = dict(ball=ball_noise, bias=ball_bias, q=q_noise, qd=qd_noise, act=act_noise, delay=delay_prob,
                          kick_p=kick_prob, kick=kick_std)
        self.render = render
        self.scene = BR.make_scene(self.mj_model, R, o) if render else None

    @property
    def observation_size(self):
        size = {"state": 84, "vis_state": 66, "privileged_state": 85}
        if self.render:
            size["pixels/view_0"] = (BR.RES, BR.RES, 6)
        return size

    @property
    def action_size(self):
        return 20

    @property
    def backend(self):
        return "mjx"

    def _init(self, rng):
        """Random start: hand pose noise, balls on the orbit at a random phase; targets start aligned."""
        r1, r2, r3, r4, r5 = jax.random.split(rng, 5)
        hand_q = jnp.clip(E.DEFAULT_POSE + 0.05 * jax.random.normal(r1, (20,)), self.lo, self.hi)
        phi = jax.random.uniform(r2, (), maxval=2 * jnp.pi)
        yz = E.targets(phi) + 0.002 * jax.random.normal(r3, (2, 2))
        p = jnp.concatenate([jnp.full((2, 1), BALL_X), yz], 1) @ self.R.T + self.o
        balls = self.q_ball.at[0:3].set(p[0]).at[7:10].set(p[1])
        data = mjx.make_data(self.mjx_model).replace(qpos=jnp.concatenate([hand_q, balls]), ctrl=hand_q)
        data = mjx.forward(self.mjx_model, data)
        b = E.balls_hand(data.qpos, self.R, self.o)
        info = {"rng": r5, "theta": phi, "pair": E.pair_angle(b), "turn": jnp.zeros(()),
                "bias": self.noise["bias"] * jax.random.normal(r4, (2, 3)),
                "delay": (jax.random.uniform(jax.random.fold_in(r4, 1)) < self.noise["delay"]).astype(jnp.float32),
                "prev_act": jnp.zeros(20), "hist": jnp.zeros((2, 2, 3)), "frame": jnp.zeros((BR.RES, BR.RES, 3), jnp.uint8)}
        return data, info

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
        prop = jnp.concatenate([q, 0.1 * qd, data.ctrl])
        balls = 10 * (jnp.concatenate([meas[None], hist]) - E.CENTER).ravel()
        priv = jnp.concatenate([E.observe(data.qpos, data.qvel, data.ctrl, th, self.R, self.o),
                                info["delay"][None], 1000 * info["bias"].ravel()])
        obs = {"state": jnp.concatenate([prop, balls, goal]), "vis_state": jnp.concatenate([prop, goal]),
               "privileged_state": priv}
        info = {**info, "hist": jnp.stack([meas, hist[0]])}
        if self.render:
            img = BR.render(data.geom_xpos, data.geom_xmat, self.scene)
            prev = jnp.where(first, img, info["frame"])
            obs["pixels/view_0"] = jnp.concatenate([img, prev], -1)
            info = {**info, "frame": img}
        return obs, info

    def reset(self, rng):
        data, info = self._init(rng)
        obs, info = self._obs(data, info, jax.random.fold_in(rng, 7), True)
        metrics = {"track_err": jnp.zeros(()), "turn": jnp.zeros(()), "dropped": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        data, info = state.pipeline_state, state.info
        # brax AutoResetWrapper restores a fixed first state and keeps info: re-sample the start ourselves
        if "steps" in info:
            fresh = info["steps"] == 0
            d0, i0 = self._init(info["rng"])
            data = jax.tree.map(lambda a, b: jnp.where(fresh, a, b), d0, data)
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

        theta = info["theta"] + self.omega * self.ctrl_dt
        b = E.balls_hand(data.qpos, self.R, self.o)
        err = jnp.linalg.norm(b[:, 1:] - E.targets(theta), axis=-1)
        pair = E.pair_angle(b)
        dpair = (pair - info["pair"] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        dropped = E.dropped_fn(b)
        # tracking + rotation progress capped at the target pace (faster earns nothing) + alive
        reward = (jnp.exp(-err / 0.015).sum() + self.w_turn * jnp.clip(dpair, -0.3, self.omega * self.ctrl_dt)
                  - 0.001 * jnp.sum(action ** 2) - 50.0 * dropped)

        info = {**info, "rng": rng, "theta": theta, "pair": pair, "turn": info["turn"] + dpair, "prev_act": act}
        obs, info = self._obs(data, info, k_obs, fresh)
        metrics = {**state.metrics, "track_err": err.mean(), "turn": dpair, "dropped": dropped.astype(jnp.float32)}
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = (dropped | nan).astype(jnp.float32)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward), done=done,
                             metrics=metrics, info=info)
