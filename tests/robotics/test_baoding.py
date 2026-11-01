import numpy as np

from robotics.policies.scripted import baoding as B


def test_balls_rest_in_hand_and_gait_in_range():
    p = (B.LOW + B.HIGH) / 2
    ang, dropped, _ = B.rollout(p, duration=1.0)
    assert not dropped and np.isfinite(ang)
    lo, hi = B.HAND_RANGE
    q = B.gait(p, 0.3)
    assert np.all(q >= lo) and np.all(q <= hi)


def test_raycast_sees_balls_at_their_projections():
    import jax.numpy as jnp
    import mujoco
    from robotics.policies.rl import baoding_env as E, baoding_render as BR
    m, R, o = E.build_model()
    d = mujoco.MjData(m)
    d.qpos[:20] = E.DEFAULT_POSE
    mujoco.mj_forward(m, d)
    sc = BR.make_scene(m, R, o)
    img = np.asarray(BR.render(jnp.array(d.geom_xpos), jnp.array(d.geom_xmat), sc))
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    for i, ch in ((0, 0), (1, 2)):  # ball0 red, ball1 blue
        v = d.body(f"ball{i}").xpos - np.asarray(sc["cam"])
        f, up, right = -R[:, 0], R[:, 2], R[:, 1]
        k = np.tan(BR.FOV / 2) * (v @ f)
        col, row = int((v @ right / k + 1) / 2 * 64), int((1 - v @ up / k) / 2 * 64)
        px = img[row, col].astype(int)
        assert px[ch] > px.sum() - px[ch], (i, px)
