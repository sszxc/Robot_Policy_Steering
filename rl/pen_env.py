"""MJX env: palm-up HX5 hand rotates a pen lying across the palm about the palm normal, in a commanded direction."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from sim import hand as H, scene

HAND_POS = (0.0, 0.0, 0.3)          # palm up, fingers along +x; palm surface at z~0.321
PEN_R, PEN_HALF = 0.005, 0.065      # capsule radius and half-length of the cylinder part (14 cm tip to tip)
# Pen across the proximal phalanges (on the palm, fingertips can't push its sides; it can't be rotated).
PEN_START = np.array([0.12, 0.0, 0.33])
CRADLE = (0, 0.2, 0.6, 0.36)          # slightly curled fingers hold the pen from rolling back onto the palm
DEFAULT_POSE = H.pose(thumb=(0, -0.5, 0.3, 0.2), index=CRADLE, middle=CRADLE, ring=CRADLE, pinky=CRADLE)


def build_model(sim_dt=0.005, condim=3):
    """Palm-up hand + free capsule pen; only hand<->pen contacts. condim 3: no torsional friction on the line contact."""
    spec = scene.hand_only(pos=HAND_POS)
    b = spec.worldbody.add_body(name="pen", pos=PEN_START, quat=[0.7071068, 0.7071068, 0, 0])
    b.add_freejoint(name="pen")
    b.add_geom(name="pen", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PEN_R, PEN_HALF, 0], mass=0.015,
               condim=condim, friction=[1, 0.01, 0.001], rgba=[0.15, 0.35, 0.85, 1])
    b.add_geom(name="pen_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PEN_R + 0.0005, 0.012, 0],
               pos=[0, 0, PEN_HALF - 0.012], contype=0, conaffinity=0, density=0, rgba=[0.9, 0.9, 0.9, 1])
    m = spec.compile()
    b = m.body("pen").id  # 15 g, 5 mm radius: roll inertia ~2e-7 is unstable in MJX at 5 ms / 5 iters
    m.body_inertia[b] = np.maximum(m.body_inertia[b], 2e-6)
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if m.geom(g).name == "pen":
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
        elif body.startswith(("finger_r_", "hand_base")) and (m.geom_contype[g] or m.geom_conaffinity[g]):
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
        else:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    m.opt.timestep = sim_dt
    m.opt.iterations, m.opt.ls_iterations = 5, 8
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    return m


def pen_axis(quat):
    """Unit pen axis (capsule local z) in world frame."""
    w, x, y, z = quat
    return jnp.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def yaw(axis):
    return jnp.arctan2(axis[1], axis[0])


def observe(qpos, qvel, ctrl, direction):
    ax = pen_axis(qpos[23:27])
    obs = jnp.concatenate([qpos[:20], 0.1 * qvel[:20], ctrl, 10 * (qpos[20:23] - PEN_START), ax,
                           0.1 * qvel[20:23], 0.1 * qvel[23:26], jnp.array([direction])])
    return jnp.clip(jnp.nan_to_num(obs), -100, 100)  # a rare NaN step must not poison the obs normalizer


def dropped_fn(qpos):
    p, ax = qpos[20:23], pen_axis(qpos[23:27])
    return (jnp.linalg.norm(p[:2] - PEN_START[:2]) > 0.07) | (p[2] < PEN_START[2] - 0.03) | (jnp.abs(ax[2]) > 0.7)


class PenEnv(Env):
    def __init__(self, ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15, max_rate=3.0, w_rate=2.0, drop_penalty=10.0,
                 condim=3):
        self.mj_model = build_model(sim_dt, condim)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.n_frames = int(round(ctrl_dt / sim_dt))
        self.ctrl_dt, self.action_scale, self.max_rate = ctrl_dt, action_scale, max_rate
        self.w_rate, self.drop_penalty = w_rate, drop_penalty
        self.lo, self.hi = jnp.array(self.mj_model.actuator_ctrlrange.T)
        self.q_pen = jnp.array(self.mj_model.qpos0[20:])

    @property
    def observation_size(self):
        return 20 * 3 + 3 + 3 + 3 + 3 + 1

    @property
    def action_size(self):
        return 20

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        r1, r2, r3 = jax.random.split(rng, 3)
        hand_q = jnp.clip(DEFAULT_POSE + 0.05 * jax.random.normal(r1, (20,)), self.lo, self.hi)
        pen = self.q_pen.at[0:2].add(0.005 * jax.random.normal(r2, (2,)))
        data = mjx.make_data(self.mjx_model).replace(qpos=jnp.concatenate([hand_q, pen]), ctrl=hand_q)
        data = mjx.forward(self.mjx_model, data)
        direction = jnp.where(jax.random.bernoulli(r3), 1.0, -1.0)
        info = {"rng": rng, "dir": direction, "yaw": yaw(pen_axis(data.qpos[23:27])), "turn": jnp.zeros(())}
        obs = observe(data.qpos, data.qvel, data.ctrl, direction)
        metrics = {"turn": jnp.zeros(()), "dropped": jnp.zeros(()), "offset": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        data = state.pipeline_state
        ctrl = jnp.clip(data.ctrl + self.action_scale * action, self.lo, self.hi)
        data = data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.n_frames, lambda _, d: mjx.step(self.mjx_model, d), data)

        d = state.info["dir"]
        ax = pen_axis(data.qpos[23:27])
        new_yaw = yaw(ax)
        dyaw = (new_yaw - state.info["yaw"] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        rate = jnp.clip(d * dyaw / self.ctrl_dt, -self.max_rate, self.max_rate)
        offset = jnp.linalg.norm(data.qpos[20:22] - PEN_START[:2])
        dropped = dropped_fn(data.qpos)

        # reward: yaw rate in the commanded direction; penalize drift only beyond 3 cm, tilt, drops
        reward = (self.w_rate * rate - 5.0 * jnp.maximum(offset - 0.03, 0) - 0.5 * jnp.abs(ax[2])
                  - 0.001 * jnp.sum(action ** 2) - 2e-4 * jnp.sum(data.qvel[:20] ** 2) - self.drop_penalty * dropped)

        info = {**state.info, "yaw": new_yaw, "turn": state.info["turn"] + d * dyaw}
        metrics = {**state.metrics, "turn": d * dyaw, "dropped": dropped.astype(jnp.float32), "offset": offset}
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = (dropped | nan).astype(jnp.float32)
        obs = observe(data.qpos, data.qvel, data.ctrl, d)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward), done=done,
                             metrics=metrics, info=info)
