"""Steer SmolVLA with z: evaluate z=0 plus random z's on the same LIBERO init states.

--check instead verifies the injection on one fixed observation batch and noise: z=0 is
bit-identical to the unhooked policy, z≠0 changes actions, and each env only sees its own z.
"""
import argparse
import json

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
    parser.add_argument("--n-z", type=int, default=4, help="random z ~ N(0, I), plus z=0")
    parser.add_argument("--z-seed", type=int, default=0)
    parser.add_argument("--inits", type=int, default=2, help="init states per z")
    parser.add_argument("--init-start", type=int, default=0)
    parser.add_argument("--videos", type=int, default=1, help="save videos for the first N init states of each z")
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
    rng = np.random.default_rng(args.z_seed)
    Z = np.concatenate([np.zeros((1, dim)), rng.standard_normal((args.n_z, dim))])
    inits = list(range(args.init_start, args.init_start + args.inits))
    episodes = [(k, r) for k in range(len(Z)) for r in inits]
    n = len(episodes)
    padded = episodes + [episodes[0]] * (-n % args.batch_size)  # fill the last batch; extras dropped
    success = np.full((len(Z), len(inits)), -1)  # -1 = not run yet
    steps = np.zeros((len(Z), len(inits)), dtype=int)
    (out / "videos").mkdir()
    fps = env.metadata.get("render_fps", 20)
    for start in range(0, n, args.batch_size):
        batch = padded[start:start + args.batch_size]
        real = range(min(args.batch_size, n - start))
        set_z(Z[[k for k, _ in batch]])
        keep = [i for i in real if batch[i][1] - args.init_start < args.videos]
        res = rollout.run_batch(policy, env, procs, [r for _, r in batch], args.seed, video_envs=keep)
        for i in real:
            k, j = batch[i][0], batch[i][1] - args.init_start
            success[k, j], steps[k, j] = res["success"][i], res["steps"][i]
        for i in keep:
            k, r = batch[i]
            name = f"z{k:02d}_init{r:02d}_{'ok' if res['success'][i] else 'fail'}.mp4"
            write_video(out / "videos" / name, list(res["frames"][i]), fps)
        results = {"init_ids": inits, "z": Z.tolist(), "success": success.tolist(), "steps": steps.tolist()}
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"episodes done: {start + len(real)}/{n}")
    for k in range(len(Z)):
        print(f"z{k:02d} |z|={np.linalg.norm(Z[k]):.2f}: success {success[k].sum()}/{len(inits)}, "
              f"mean steps {steps[k].mean():.0f}")
    print(f"z=0: {success[0].mean():.2f}; random z mean: {success[1:].mean():.2f}; results in {out}")
    env.close()


if __name__ == "__main__":
    main()
