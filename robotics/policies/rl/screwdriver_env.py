"""MJX env: palm-down HX5 hand twirls a screwdriver handle (world hinge) with fingers only, in a commanded direction."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from robotics.sim import hand as H, scene

HAND_POS = (0.0, 0.0, 0.3)
AXIS = np.array([0.11, 0.02])  # handle axis (x, y); reachable by thumb, index and middle
Z0, Z1, R = 0.13, 0.21, 0.015  # handle z span and radius
# thumb, index, middle tips lightly touching the handle (found by random search on tip distance)
DEFAULT_POSE = H.pose(thumb=(0.29, -1.29, 0.58, 0.34), index=(0, 1.05, 0.65, 0.25), middle=(0, 1.1, 0.55, 0.38),
                      ring=(0, 0.53, 0.44, 0.0), pinky=(0, 0.62, 0.1, 0.3))
TIPS = ("thumb", "index", "middle", "ring")


def build_model(sim_dt=0.005):
    """Palm-down hand + screwdriver on a world hinge; only hand<->handle contacts plus hand self-collision pairs."""
    spec = scene.hand_only(pos=HAND_POS, quat=scene.hand_quat((0, 0, -1), (1, 0, 0)))
    scene.add_object(spec, "screwdriver.xml", pos=(AXIS[0], AXIS[1], Z0))
    g = spec.geom("sd_handle")  # MJX has no cylinder-box collision; capsule with the same span
    g.type, g.size = mujoco.mjtGeom.mjGEOM_CAPSULE, [R, 0.04 - R, 0]
    scene.add_hand_self_pairs(spec)
    m = spec.compile()
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if body == "screwdriver" and m.geom(g).name == "sd_handle":
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
        elif body.startswith(("finger_r_", "hand_base")) and (m.geom_contype[g] or m.geom_conaffinity[g]):
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
        else:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    m.opt.timestep = sim_dt
    m.opt.iterations, m.opt.ls_iterations = 5, 8
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    return m


def tip_dist(tips):
    """Distance (m) from fingertip points (n,3) to the handle surface, 0 when touching/inside."""
    r = jnp.hypot(tips[:, 0] - AXIS[0], tips[:, 1] - AXIS[1]) - R
    dz = jnp.maximum(tips[:, 2] - Z1, 0) + jnp.maximum(Z0 - tips[:, 2], 0)
    return jnp.hypot(jnp.maximum(r, 0), dz)


def observe(qpos, qvel, ctrl, direction):
    """Privileged obs; works on jnp or np arrays (also used by CPU eval)."""
    a = qpos[20]
    return jnp.concatenate([qpos[:20], 0.1 * qvel[:20], ctrl,
                            jnp.array([jnp.sin(a), jnp.cos(a), 0.2 * qvel[20], direction])])


class ScrewdriverEnv(Env):
    def __init__(self, direction=0, ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15, max_speed=4.0):
        """direction: +1 / -1 fixed, or 0 = random per episode."""
        self.mj_model = build_model(sim_dt)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.n_frames = int(round(ctrl_dt / sim_dt))
        self.ctrl_dt, self.direction = ctrl_dt, direction
        self.action_scale, self.max_speed = action_scale, max_speed
        self.lo, self.hi = jnp.array(self.mj_model.actuator_ctrlrange.T)
        self.tip_ids = jnp.array([self.mj_model.site(f + "_tip").id for f in TIPS])

    @property
    def observation_size(self):
        return 20 * 3 + 4

    @property
    def action_size(self):
        return 20

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        r1, r2, r3 = jax.random.split(rng, 3)
        hand_q = jnp.clip(DEFAULT_POSE + 0.05 * jax.random.normal(r1, (20,)), self.lo, self.hi)
        a0 = jax.random.uniform(r2, (), minval=-jnp.pi, maxval=jnp.pi)
        qpos = jnp.concatenate([hand_q, a0[None]])
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, ctrl=hand_q)
        data = mjx.forward(self.mjx_model, data)
        direction = jnp.where(self.direction == 0, jnp.sign(jax.random.uniform(r3) - 0.5), float(self.direction))
        info = {"rng": rng, "direction": direction}
        obs = observe(data.qpos, data.qvel, data.ctrl, direction)
        metrics = {"speed": jnp.zeros(()), "tip_dist": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        data = state.pipeline_state
        a_prev = data.qpos[20]
        ctrl = jnp.clip(data.ctrl + self.action_scale * action, self.lo, self.hi)
        data = data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.n_frames, lambda _, d: mjx.step(self.mjx_model, d), data)

        direction = state.info["direction"]
        speed = direction * (data.qpos[20] - a_prev) / self.ctrl_dt  # rad/s in commanded direction
        dist = tip_dist(data.site_xpos[self.tip_ids])
        # reward: clipped handle speed, fingertips near the handle, small action/velocity penalties
        reward = (jnp.clip(speed, -self.max_speed, self.max_speed) - 10.0 * jnp.mean(dist[:3])
                  - 0.002 * jnp.sum(action ** 2) - 1e-4 * jnp.sum(data.qvel[:20] ** 2))
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        metrics = {**state.metrics, "speed": speed, "tip_dist": jnp.mean(dist[:3])}
        obs = observe(data.qpos, data.qvel, data.ctrl, direction)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward),
                             done=nan.astype(jnp.float32), metrics=metrics)
