"""MJX env: palm-up HX5 hand holds a free Rubik's cube and turns its top layer by quarter turns (brax Env API).

Same pattern as rl/cube_env.py. The scripted version (tasks/rubik.py) needs a table fixture; this is the in-hand one.
"""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from rl.cube_env import quat_angle
from sim import hand as H, scene
from tasks.rubik import DETENT, EDGE, add_rubik

HAND_POS = (0.0, 0.0, 0.3)
CUBE_START = np.array([0.075, 0.0, 0.3 + 0.021 + EDGE / 2 + 0.001])  # resting on the palm
DEFAULT_POSE = H.pose(thumb=(0.6, -0.6, 0.4, 0.3), index=(0, 0.4, 0.3, 0.2), middle=(0, 0.4, 0.3, 0.2),
                      ring=(0, 0.4, 0.3, 0.2), pinky=(0, 0.4, 0.3, 0.2))
QUARTER = jnp.pi / 2
# qpos: hand 0:20, cube free joint 20:27, top-layer hinge 27; qvel: hand 0:20, cube 20:26, hinge 26


def build_model(sim_dt=0.005, stickers=False):
    """Hand-only palm-up scene + free Rubik's cube; only hand<->cube contacts are enabled.
    Stickers (~130 visual geoms) are only needed for rendering."""
    spec = scene.hand_only(pos=HAND_POS)
    add_rubik(spec, CUBE_START, stickers=stickers)
    m = spec.compile()
    # Only hand<->cube contacts; the top layer skips the hand's boxes (palm, knuckles): box-box is slow in MJX.
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            body = m.body(m.geom_bodyid[g]).name
            if body.startswith("rubik"):
                m.geom_contype[g], m.geom_conaffinity[g] = (2 if body == "rubik_top" else 1), 0
            elif body == "world":
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 1 if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 3
    m.opt.timestep = sim_dt
    m.opt.iterations, m.opt.ls_iterations = 5, 8
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    return m


def observe(qpos, qvel, ctrl, target):
    """Privileged obs; works on jnp or np arrays."""
    err = qpos[27] - target
    return jnp.concatenate([qpos[:20], 0.1 * qvel[:20], ctrl, (qpos[20:23] - CUBE_START) * 10, qpos[23:27],
                            qvel[20:23], 0.2 * qvel[23:26], jnp.array([jnp.sin(err), jnp.cos(err), 0.2 * qvel[26]])])


class RubikEnv(Env):
    def __init__(self, ctrl_dt=0.05, sim_dt=0.005, action_scale=0.15, success_tol=np.radians(10)):
        self.mj_model = build_model(sim_dt)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.n_frames = int(round(ctrl_dt / sim_dt))
        self.action_scale, self.success_tol = action_scale, success_tol
        self.lo, self.hi = jnp.array(self.mj_model.actuator_ctrlrange.T)

    @property
    def observation_size(self):
        return 20 * 3 + 3 + 4 + 6 + 3

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
        qpos = jnp.concatenate([hand_q, cube_pos, jnp.array([1.0, 0, 0, 0, 0])])
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, ctrl=hand_q)
        data = mjx.forward(self.mjx_model, data)
        sign = jnp.where(jax.random.bernoulli(r3), 1.0, -1.0)  # turn direction for this episode
        target = sign * QUARTER
        info = {"rng": rng, "sign": sign, "target": target, "successes": jnp.zeros(()),
                "last_err": jnp.abs(target)}
        obs = observe(data.qpos, data.qvel, data.ctrl, target)
        metrics = {"successes": jnp.zeros(()), "layer_err": jnp.zeros(()), "dropped": jnp.zeros(()),
                   "cube_rot": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _physics(self, data):
        def one(_, d):  # 4-fold detent on the hinge, like tasks/rubik.add_detent
            return mjx.step(self.mjx_model, d.replace(qfrc_applied=d.qfrc_applied.at[26].set(
                -DETENT * jnp.sin(4 * d.qpos[27]))))
        return jax.lax.fori_loop(0, self.n_frames, one, data)

    def step(self, state, action):
        data = state.pipeline_state
        ctrl = jnp.clip(data.ctrl + self.action_scale * action, self.lo, self.hi)
        data = self._physics(data.replace(ctrl=ctrl))

        target = state.info["target"]
        err = jnp.abs(data.qpos[27] - target)
        dist = jnp.linalg.norm(data.qpos[20:23] - CUBE_START)
        cube_rot = quat_angle(data.qpos[23:27], jnp.array([1.0, 0, 0, 0]))  # lower body vs start
        dropped = (dist > 0.1) | (data.qpos[22] < CUBE_START[2] - 0.04)
        success = (err < self.success_tol) & (jnp.abs(data.qvel[26]) < 1.0)

        # reward: layer progress toward target, keep the cube still in the palm, success bonus, drop penalty
        reward = (20.0 * (state.info["last_err"] - err) + 0.5 / (err + 0.1) - 10.0 * dist - 0.5 * cube_rot
                  - 0.001 * jnp.sum(action ** 2) + 100.0 * success - 100.0 * dropped)

        target = jnp.where(success, target + state.info["sign"] * QUARTER, target)
        info = {**state.info, "target": target, "successes": state.info["successes"] + success,
                "last_err": jnp.abs(data.qpos[27] - target)}
        metrics = {**state.metrics, "successes": success.astype(jnp.float32), "layer_err": err,
                   "dropped": dropped.astype(jnp.float32), "cube_rot": cube_rot}
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = (dropped | nan).astype(jnp.float32)
        obs = observe(data.qpos, data.qvel, data.ctrl, target)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.where(nan, 0.0, reward), done=done,
                             metrics=metrics, info=info)
