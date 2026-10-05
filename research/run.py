"""Steer SmolVLA with z: z=0 (noise reps) and scale * random directions on the same init states.

--check instead verifies the injection on one fixed observation batch and noise: z=0 is
bit-identical to the unhooked policy, z≠0 changes actions, and each env only sees its own z.
"""
import argparse

import numpy as np
import torch

from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.utils.io_utils import write_video
from research import rollout
from research.inject import INJECT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    rollout.add_args(parser)
    parser.add_argument("--inject", default="expert_film", choices=INJECT)
    parser.add_argument("--layers", default="7,15,23,31")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--w", default="random", help='"random" or a fit_pca.py pca.pt path')
    parser.add_argument("--w-seed", type=int, default=0)
    parser.add_argument("--n-z", type=int, default=4, help="orthonormal random directions u_k (≤ dim), ‖u_k‖ = √dim")
    parser.add_argument("--scales", default="1", help="comma list; z = scale * u for each u")
    parser.add_argument("--zero-reps", type=int, default=1, help="z=0 episodes per init (noise reps); 0 skips z=0")
    parser.add_argument("--z-reps", type=int, default=1, help="noise reps per init for z≠0")
    parser.add_argument("--z-seed", type=int, default=0)
    parser.add_argument("--z-file", default=None, help=".npy [K, dim]: use these as the directions u_k")
    parser.add_argument("--inits", type=int, default=2, help="init states per z")
    parser.add_argument("--init-start", type=int, default=0)
    parser.add_argument("--videos", type=int, default=1, help="save rep-0 videos for the first N init states of each z")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def check(args, policy, env, procs, attach):
    B = args.batch_size
    assert B >= 2, "--check needs --batch-size >= 2"
    env.set_attr("init_state_id", list(range(B)))
    obs, _ = env.reset(seed=[args.seed + i for i in range(B)], options={NEW_ROLLOUT_OPTION: True})
    obs = preprocess_observation(obs)
    obs["task"] = list(env.call("task_description"))
    env_pre, _, pre, _ = procs
    batch = pre(env_pre(obs))
    cfg = policy.config
    noise = torch.randn(B, cfg.chunk_size, cfg.max_action_dim, device="cuda",
                        generator=torch.Generator("cuda").manual_seed(args.seed))

    def actions(z=None):  # same observations and flow-matching noise every call
        if z is not None:
            set_z(z)
        policy.reset()
        return policy.predict_action_chunk(dict(batch), noise=noise.clone())

    base = actions()
    dim, set_z = attach()
    z1 = torch.randn(dim, generator=torch.Generator().manual_seed(args.z_seed))
    zero = torch.zeros(B, dim)
    ok = torch.equal(actions(zero), base)
    print(f"z=0 equals unhooked policy: {ok}")
    for i in (0, B - 1):
        z = zero.clone()
        z[i] = z1
        a = actions(z)
        others = [j for j in range(B) if j != i]
        same, changed = torch.equal(a[others], base[others]), not torch.equal(a[i], base[i])
        print(f"z1 on env {i}: env {i} changed {changed}, other envs unchanged {same}")
        ok &= same and changed
    for c in (0.5, 1, 2, 4, 8):
        d = (actions(c * z1.expand(B, dim)) - base).abs().mean() / base.abs().mean()
        print(f"z = {c} * z1 (|z|={c * z1.norm():.2f}): action chunk mean |Δa| / mean |a| = {d:.3f}")
    print("CHECK PASSED" if ok else "CHECK FAILED")


def directions(n, dim, seed):
    """n ≤ dim orthonormal random directions [n, dim], each scaled to norm √dim (≈ ‖N(0, I)‖)."""
    assert n <= dim, "at most dim orthonormal directions"
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return (q * np.sign(np.diag(r))).T[:n] * dim**0.5


def plan(args, dim):
    """Z rows: 0 = zero, then scale * u_k for each scale and direction u_k (see directions).
    Episodes (z row, init, rep): z=0 for zero_reps reps, others z_reps; noise is paired by (init, rep)."""
    U = np.load(args.z_file) if args.z_file else directions(args.n_z, dim, args.z_seed)
    args.n_z = len(U)
    scales = [float(c) for c in args.scales.split(",")]
    Z = np.concatenate([np.zeros((1, dim))] + [c * U for c in scales])
    scale = np.array([0.0] + [c for c in scales for _ in range(args.n_z)])
    direction = np.array([-1] + [k for _ in scales for k in range(args.n_z)])
    inits = range(args.init_start, args.init_start + args.inits)
    episodes = [(0, r, j) for r in inits for j in range(args.zero_reps)]
    episodes += [(k, r, j) for k in range(1, len(Z)) for r in inits for j in range(args.z_reps)]
    return Z, scale, direction, episodes


def main():
    args = parse_args()
    out = rollout.new_run_dir(args, f"steer-{args.suite}-t{args.task_id}")
    policy, env, procs = rollout.load(args)
    attach = lambda: INJECT[args.inject](policy, layers=[int(l) for l in args.layers.split(",")],
                                         dim=args.dim, alpha=args.alpha, w=args.w, seed=args.w_seed)
    if args.check:
        check(args, policy, env, procs, attach)
        return env.close()

    dim, set_z = attach()
    Z, scale, direction, episodes = plan(args, dim)
    n, B, T = len(episodes), args.batch_size, env.call("_max_episode_steps")[0]
    padded = episodes + [episodes[0]] * (-n % B)  # fill the last batch; extras dropped
    ep = np.array(episodes)
    res_all = {"success": np.zeros(n, bool), "steps": np.zeros(n, int),
               "state": np.zeros((n, T, 8), np.float32), "action": np.zeros((n, T, 7), np.float32)}
    (out / "videos").mkdir()
    fps = env.metadata.get("render_fps", 20)
    for start in range(0, n, B):
        batch = padded[start:start + B]
        real = range(min(B, n - start))
        set_z(Z[[k for k, _, _ in batch]])
        keep = [i for i in real if batch[i][1] - args.init_start < args.videos and batch[i][2] == 0]
        res = rollout.run_batch(policy, env, procs, [r for _, r, _ in batch], args.seed,
                                noise_seeds=[args.seed + r + 1000 * j for _, r, j in batch], video_envs=keep)
        for i in real:
            e, t = start + i, res["state"].shape[1]
            res_all["success"][e], res_all["steps"][e] = res["success"][i], res["steps"][i]
            # after done the env is frozen; pad with the last valid value
            res_all["state"][e, :t] = res["state"][i]
            res_all["state"][e, t:] = res["state"][i, -1]
            res_all["action"][e, :t] = res["action"][i]
        for i in keep:
            k, r, _ = batch[i]
            name = f"z{k:02d}_s{scale[k]:+.2f}_d{direction[k]}_init{r:02d}_{'ok' if res['success'][i] else 'fail'}.mp4"
            write_video(out / "videos" / name, list(res["frames"][i]), fps)
        np.savez_compressed(out / "results.npz", Z=Z, scale=scale, direction=direction, ep_z=ep[:, 0],
                            ep_init=ep[:, 1], ep_rep=ep[:, 2], done=np.arange(n) < start + len(real), **res_all)
        print(f"episodes done: {start + len(real)}/{n}", flush=True)
    sr = {}
    for k in range(len(Z)):
        m = ep[:, 0] == k
        sr.setdefault(scale[k], []).append(res_all["success"][m].mean())
    for c, v in sr.items():
        print(f"scale {c:+.2f}: success {np.mean(v):.3f} over {len(v)} z")
    print(f"results in {out}")
    env.close()


if __name__ == "__main__":
    main()
