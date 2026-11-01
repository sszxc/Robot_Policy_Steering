"""Pure-JAX ray caster for the Baoding scene: top-down RGB of the collision primitives (sphere/capsule/box), vmappable in MJX."""
import jax.numpy as jnp
import mujoco
import numpy as np

RES = 64
LOOK = np.array([0.043, 0.0, 0.09])   # view center in the hand frame (between palm and fingers)
DIST, FOV = 0.25, np.radians(34)      # camera height above LOOK along the palm normal, full field of view
BG = np.array([0.15, 0.17, 0.2])
HAND_RGB = {mujoco.mjtGeom.mjGEOM_CAPSULE: (0.82, 0.78, 0.72), mujoco.mjtGeom.mjGEOM_BOX: (0.55, 0.56, 0.62)}


def make_scene(m, R, o, res=RES):
    """Static camera rays + geom tables. R, o: hand frame (camera looks along -palm normal, image up = fingers)."""
    f, up, right = -R[:, 0], R[:, 2], R[:, 1]
    c = o + R @ LOOK - DIST * f
    v, u = np.meshgrid(1 - (np.arange(res) + 0.5) / res * 2, (np.arange(res) + 0.5) / res * 2 - 1, indexing="ij")
    d = f + np.tan(FOV / 2) * (u[..., None] * right + v[..., None] * up)
    d = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)
    light = -f + 0.5 * up + 0.3 * right
    groups = {}
    for t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX):
        ids = [g for g in range(m.ngeom) if m.geom_type[g] == t and m.geom_bodyid[g] != 0]
        rgb = [m.geom_rgba[g, :3] if t == mujoco.mjtGeom.mjGEOM_SPHERE else HAND_RGB[t] for g in ids]
        groups[t] = (np.array(ids), jnp.array(m.geom_size[ids]), jnp.array(np.array(rgb)))
    return dict(cam=jnp.array(c), rays=jnp.array(d), light=jnp.array(light / np.linalg.norm(light)),
                groups=groups, res=res)


def _sphere_t(oc, d, r):
    """oc: (..., 3) camera minus center, d: (P, 3). Nearest positive hit t or inf."""
    b = jnp.sum(oc * d, -1)
    disc = b ** 2 - (jnp.sum(oc ** 2, -1) - r ** 2)
    t = -b - jnp.sqrt(jnp.maximum(disc, 0))
    return jnp.where((disc > 0) & (t > 0), t, jnp.inf)


def _spheres(pos, size, sc):
    oc = (sc["cam"] - pos)[:, None]                                   # (G, 1, 3)
    t = _sphere_t(oc, sc["rays"][None], size[:, :1])
    n = (oc + t[..., None] * sc["rays"][None]) / size[:, None, :1]
    return t, n @ sc["light"]


def _capsules(pos, mat, size, sc):
    ol = jnp.einsum("gji,gj->gi", mat, sc["cam"] - pos)[:, None]      # camera in geom frame (G, 1, 3)
    dl = jnp.einsum("gji,pj->gpi", mat, sc["rays"])                    # (G, P, 3)
    ll = jnp.einsum("gji,j->gi", mat, sc["light"])[:, None]
    r, h = size[:, None, 0], size[:, None, 1]
    a = dl[..., 0] ** 2 + dl[..., 1] ** 2
    b = ol[..., 0] * dl[..., 0] + ol[..., 1] * dl[..., 1]
    disc = b ** 2 - a * (ol[..., 0] ** 2 + ol[..., 1] ** 2 - r ** 2)
    tc = (-b - jnp.sqrt(jnp.maximum(disc, 0))) / jnp.maximum(a, 1e-9)
    zc = ol[..., 2] + tc * dl[..., 2]
    tc = jnp.where((disc > 0) & (tc > 0) & (jnp.abs(zc) <= h), tc, jnp.inf)
    nc = jnp.stack([ol[..., 0] + tc * dl[..., 0], ol[..., 1] + tc * dl[..., 1], 0 * tc], -1) / r[..., None]
    best_t, best_n = tc, nc
    for s in (1.0, -1.0):
        oc = ol - jnp.stack([0 * h, 0 * h, s * h], -1)
        ts = _sphere_t(oc, dl, r)
        ns = (oc + ts[..., None] * dl) / r[..., None]
        best_n = jnp.where((ts < best_t)[..., None], ns, best_n)
        best_t = jnp.minimum(ts, best_t)
    return best_t, jnp.sum(jnp.nan_to_num(best_n) * ll, -1)


def _boxes(pos, mat, size, sc):
    ol = jnp.einsum("gji,gj->gi", mat, sc["cam"] - pos)[:, None]
    dl = jnp.einsum("gji,pj->gpi", mat, sc["rays"])
    dl = jnp.where(jnp.abs(dl) < 1e-9, 1e-9, dl)
    ll = jnp.einsum("gji,j->gi", mat, sc["light"])[:, None]
    s = size[:, None]
    t1, t2 = (-s - ol) / dl, (s - ol) / dl
    tn, tf = jnp.minimum(t1, t2), jnp.maximum(t1, t2)
    tmin, tmax = tn.max(-1), tf.min(-1)
    t = jnp.where((tmax >= tmin) & (tmin > 0), tmin, jnp.inf)
    axis = jnp.argmax(tn, -1)
    sign = -jnp.sign(jnp.take_along_axis(dl, axis[..., None], -1)[..., 0])
    return t, sign * jnp.take_along_axis(jnp.broadcast_to(ll, dl.shape), axis[..., None], -1)[..., 0]


def render(geom_xpos, geom_xmat, sc):
    """One (res, res, 3) uint8 image from an (mjx or mj) data's geom poses."""
    mat = geom_xmat.reshape(-1, 3, 3)
    ts, shades, rgbs = [], [], []
    for t, (ids, size, rgb) in sc["groups"].items():
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            tt, sh = _spheres(geom_xpos[ids], size, sc)
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            tt, sh = _capsules(geom_xpos[ids], mat[ids], size, sc)
        else:
            tt, sh = _boxes(geom_xpos[ids], mat[ids], size, sc)
        ts.append(tt), shades.append(sh), rgbs.append(rgb)
    t, sh, rgb = jnp.concatenate(ts), jnp.concatenate(shades), jnp.concatenate(rgbs)
    k = jnp.argmin(t, 0)
    hit = jnp.isfinite(jnp.take_along_axis(t, k[None], 0)[0])
    shade = 0.35 + 0.65 * jnp.clip(jnp.take_along_axis(sh, k[None], 0)[0], 0, 1)
    img = jnp.where(hit[:, None], rgb[k] * shade[:, None], BG)
    return (255 * img).astype(jnp.uint8).reshape(sc["res"], sc["res"], 3)
