"""MJX env: tilted palm-up HX5 hand orbits two Baoding balls by tracking targets rotating about a palm point."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from robotics.sim import hand as H, scene
from robotics.policies.scripted import baoding as B

TILT = (-0.3, 0.0)  # (pitch, roll) of the palm: fingers slightly downhill so the balls rest against them
DEFAULT_POSE = H.pose(thumb=(0, -0.2, 0.8, 0.5), index=(0, 1.2, 0.5, 0.35), middle=(0, 1.2, 0.5, 0.35),
                      ring=(0, 1.2, 0.5, 0.35), pinky=(0, 1.2, 0.5, 0.35))
CENTER = np.array([0.043, 0.0, 0.086])  # orbit center in the hand frame (balls rest at z~0.108 against the fingers)
RADIUS = 0.022                           # ball0 starts at orbit angle 0 (+y side)
PALM_X = 0.021                                            # palm surface height in the hand frame


def build_model(sim_dt=0.005, r=B.R, mass=B.MASS):
    """Hand + balls; hand<->ball, ball<->ball and hand self-collision pairs. Returns (model, R_hand, o_hand)."""
    spec = B.ball_spec(*TILT, solref=(0.02, 1), r=r, mass=mass)
    scene.add_hand_self_pairs(spec)
    m = spec.compile()
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if body.startswith("ball"):
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 1
        elif body == "world":
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
        elif m.geom_contype[g] or m.geom_conaffinity[g]:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
    m.opt.timestep = sim_dt
    m.opt.iterations, m.opt.ls_iterations = 5, 8
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return m, d.body("hand_base").xmat.reshape(3, 3).copy(), d.body("hand_base").xpos.copy()


def balls_hand(qpos, R, o):
    """Ball centers (2, 3) in the hand frame."""
    return (jnp.stack([qpos[20:23], qpos[27:30]]) - o) @ R


def targets(theta, c=CENTER[1:], rad=RADIUS):
    """Target (y, z) of both balls for orbit angle theta (orbit center c = (y, z), radius rad); ball1 opposite ball0."""
    ang = jnp.array([theta, theta + jnp.pi])
    return c + rad * jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)


def pair_angle(b):
    return jnp.arctan2(b[1, 2] - b[0, 2], b[1, 1] - b[0, 1])


def observe(qpos, qvel, ctrl, theta, R, o, tg=None):
    """Positions relative to the fixed reference CENTER; tg = target (y, z) of both balls (default: targets(theta))."""
    b = balls_hand(qpos, R, o)
    v = jnp.stack([qvel[20:23], qvel[26:29]]) @ R
    tg = targets(theta) if tg is None else tg
    return jnp.concatenate([qpos[:20], 0.1 * qvel[:20], ctrl, 10 * (b - CENTER).ravel(), v.ravel(),
                            10 * (tg - CENTER[1:]).ravel(), jnp.array([jnp.sin(theta), jnp.cos(theta)])])


def dropped_fn(b, r=B.R):
    return ((b[:, 0] < PALM_X) | (b[:, 0] > PALM_X + r + 0.015) | (jnp.abs(b[:, 1]) > 0.09)
            | (b[:, 2] < -0.02) | (b[:, 2] > 0.2)).any()


class BaodingEnv(Env):
    def __init__(self, period=8.0, direction=1.0, ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15, w_turn=10.0):
        self.mj_model, R, o = build_model(sim_dt)
        self.R, self.o = jnp.array(R), jnp.array(o)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.n_frames = int(round(ctrl_dt / sim_dt))
        self.omega = direction * 2 * np.pi / period
        self.ctrl_dt, self.action_scale, self.w_turn = ctrl_dt, action_scale, w_turn
        self.lo, self.hi = jnp.array(self.mj_model.actuator_ctrlrange.T)
        self.q_ball = jnp.array(self.mj_model.qpos0[20:])

    @property
    def observation_size(self):
        return 20 * 3 + 6 + 6 + 4 + 2

    @property
    def action_size(self):
        return 20

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        r1, r2 = jax.random.split(rng)
        hand_q = jnp.clip(DEFAULT_POSE + 0.05 * jax.random.normal(r1, (20,)), self.lo, self.hi)
        noise = 0.003 * jax.random.normal(r2, (2, 3)) * jnp.array([0, 1, 1])
        balls = self.q_ball.at[0:3].add(noise[0] @ self.R.T).at[7:10].add(noise[1] @ self.R.T)
        data = mjx.make_data(self.mjx_model).replace(qpos=jnp.concatenate([hand_q, balls]), ctrl=hand_q)
        data = mjx.forward(self.mjx_model, data)
        b = balls_hand(data.qpos, self.R, self.o)
        info = {"rng": rng, "theta": jnp.zeros(()), "pair": pair_angle(b), "turn": jnp.zeros(())}
        obs = observe(data.qpos, data.qvel, data.ctrl, 0.0, self.R, self.o)
        metrics = {"track_err": jnp.zeros(()), "turn": jnp.zeros(()), "dropped": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        data = state.pipeline_state
        ctrl = jnp.clip(data.ctrl + self.action_scale * action, self.lo, self.hi)
        data = data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.n_frames, lambda _, d: mjx.step(self.mjx_model, d), data)

        theta = state.info["theta"] + self.omega * self.ctrl_dt
        b = balls_hand(data.qpos, self.R, self.o)
        err = jnp.linalg.norm(b[:, 1:] - targets(theta), axis=-1)
        pair = pair_angle(b)
        dpair = jnp.clip((pair - state.info["pair"] + jnp.pi) % (2 * jnp.pi) - jnp.pi, -0.3, 0.3)
        dropped = dropped_fn(b)

        # reward: track the rotating targets, progress of the pair axis in the commanded direction, stay alive
        reward = (jnp.exp(-err / 0.015).sum() + self.w_turn * dpair * jnp.sign(self.omega)
                  - 0.001 * jnp.sum(action ** 2) - 50.0 * dropped)

        info = {**state.info, "theta": theta, "pair": pair, "turn": state.info["turn"] + dpair}
        metrics = {**state.metrics, "track_err": err.mean(), "turn": dpair, "dropped": dropped.astype(jnp.float32)}
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = (dropped | nan).astype(jnp.float32)
        obs = observe(data.qpos, data.qvel, data.ctrl, theta, self.R, self.o)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward), done=done,
                             metrics=metrics, info=info)
