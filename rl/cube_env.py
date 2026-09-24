"""MJX env: palm-up HX5 hand reorients a cube to random goal orientations (brax Env API)."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from sim import hand as H, scene

HAND_POS = (0.0, 0.0, 0.3)
CUBE_START = np.array([0.092, 0.0, 0.358])  # resting on the palm
DEFAULT_POSE = H.pose(thumb=(0.6, -0.6, 0.4, 0.3), index=(0, 0.4, 0.3, 0.2), middle=(0, 0.4, 0.3, 0.2),
                      ring=(0, 0.4, 0.3, 0.2), pinky=(0, 0.4, 0.3, 0.2))


def build_model(sim_dt=0.005, goal_marker=False):
    """Hand-only palm-up scene + cube; only hand<->cube contacts plus explicit hand self-collision pairs."""
    spec = scene.hand_only(pos=HAND_POS)
    scene.add_object(spec, "rl_cube.xml", pos=CUBE_START)
    if goal_marker:  # mocap ghost cube showing the goal orientation (adds no qpos)
        ghost = scene.add_object(spec, "rl_cube.xml", pos=CUBE_START + [0, 0.12, 0.05], prefix="goal_")
        ghost.mocap = True
        spec.delete(ghost.first_joint())
        for g in ghost.geoms:
            g.contype = g.conaffinity = 0
            g.rgba[3] = 0.5
    scene.add_hand_self_pairs(spec)
    m = spec.compile()
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            if body == "cube":
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
            elif body == "world":
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
    m.opt.timestep = sim_dt
    m.opt.iterations, m.opt.ls_iterations = 5, 8
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    return m


def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def quat_conj(q):
    return q * jnp.array([1.0, -1, -1, -1])


def quat_angle(a, b):
    """Rotation angle (rad) between two unit quaternions."""
    return 2 * jnp.arcsin(jnp.clip(jnp.linalg.norm(quat_mul(a, quat_conj(b))[..., 1:], axis=-1), 0, 1))


def sample_goal(rng, mode):
    if mode == "z":  # yaw about world z only (curriculum stage)
        a = jax.random.uniform(rng, (), minval=-jnp.pi, maxval=jnp.pi)
        return jnp.array([jnp.cos(a / 2), 0, 0, jnp.sin(a / 2)])
    q = jax.random.normal(rng, (4,))
    return q / jnp.linalg.norm(q)


def observe(qpos, qvel, ctrl, goal):
    """Privileged obs; works on jnp or np arrays (also used by CPU eval)."""
    cube_pos, cube_quat = qpos[20:23], qpos[23:27]
    return jnp.concatenate([qpos[:20], 0.1 * qvel[:20], ctrl, (cube_pos - CUBE_START) * 10, cube_quat,
                            qvel[20:23], 0.2 * qvel[23:26], goal, quat_mul(goal, quat_conj(cube_quat))])


class CubeEnv(Env):
    def __init__(self, goal_mode="full", ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15, success_tol=0.4):
        self.mj_model = build_model(sim_dt)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.n_frames = int(round(ctrl_dt / sim_dt))
        self.goal_mode, self.action_scale, self.success_tol = goal_mode, action_scale, success_tol
        self.lo, self.hi = jnp.array(self.mj_model.actuator_ctrlrange.T)

    @property
    def observation_size(self):
        return 20 * 3 + 3 + 4 + 6 + 4 + 4

    @property
    def action_size(self):
        return 20

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        r1, r2, r3 = jax.random.split(rng, 3)
        hand_q = jnp.clip(DEFAULT_POSE + 0.05 * jax.random.normal(r1, (20,)), self.lo, self.hi)
        cube_pos = CUBE_START + jnp.array([0.005, 0.005, 0.0]) * jax.random.normal(r2, (3,))
        qpos = jnp.concatenate([hand_q, cube_pos, jnp.array([1.0, 0, 0, 0])])
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, ctrl=hand_q)
        data = mjx.forward(self.mjx_model, data)
        goal = sample_goal(r3, self.goal_mode)
        info = {"rng": rng, "goal": goal, "successes": jnp.zeros(()), "last_ang": quat_angle(goal, qpos[23:27])}
        obs = observe(data.qpos, data.qvel, data.ctrl, goal)
        metrics = {"successes": jnp.zeros(()), "ang_err": jnp.zeros(()), "dropped": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        data = state.pipeline_state
        ctrl = jnp.clip(data.ctrl + self.action_scale * action, self.lo, self.hi)
        data = data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.n_frames, lambda _, d: mjx.step(self.mjx_model, d), data)

        goal = state.info["goal"]
        cube_pos, cube_quat = data.qpos[20:23], data.qpos[23:27]
        ang = quat_angle(goal, cube_quat)
        dist = jnp.linalg.norm(cube_pos - CUBE_START)
        dropped = (dist > 0.1) | (cube_pos[2] < CUBE_START[2] - 0.04)
        success = ang < self.success_tol

        # reward: inverse orientation error + progress, keep cube near palm, success bonus, drop penalty
        reward = (1.0 / (ang + 0.1) + 5.0 * (state.info["last_ang"] - ang) - 10.0 * dist
                  - 0.001 * jnp.sum(action ** 2) + 250.0 * success - 100.0 * dropped)

        rng, r_goal = jax.random.split(state.info["rng"])
        goal = jnp.where(success, sample_goal(r_goal, self.goal_mode), goal)
        # wrappers add their own keys, so update in place
        info = {**state.info, "rng": rng, "goal": goal, "successes": state.info["successes"] + success,
                "last_ang": quat_angle(goal, cube_quat)}
        metrics = {**state.metrics, "successes": success.astype(jnp.float32), "ang_err": ang,
                   "dropped": dropped.astype(jnp.float32)}
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = (dropped | nan).astype(jnp.float32)
        obs = observe(data.qpos, data.qvel, data.ctrl, goal)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward), done=done,
                             metrics=metrics, info=info)
