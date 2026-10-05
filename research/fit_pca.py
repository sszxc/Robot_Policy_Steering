"""PCA of action-expert MLP outputs under the unmodified policy (input for expert_film --w PATH).

Every action token of every denoising step is one H-dim sample; nothing is grouped by τ yet.
"""
import argparse

import torch

from research import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    rollout.add_args(parser)
    parser.add_argument("--layers", default="7,15,23,31")
    parser.add_argument("--episodes", type=int, default=24, help="init states 0..episodes-1")
    parser.add_argument("--obs-every", type=int, default=20, help="keep policy inputs every N steps (openloop.py)")
    args = parser.parse_args()
    assert args.episodes % args.batch_size == 0, "episodes must be a multiple of batch size"
    layers = [int(l) for l in args.layers.split(",")]
    out = rollout.new_run_dir(args, f"pca-{args.suite}-t{args.task_id}")
    policy, env, procs = rollout.load(args)

    expert = policy.model.vlm_with_expert.lm_expert
    stats = {l: [0, 0.0, 0.0] for l in layers}  # n, Σx, Σxxᵀ (float64)

    def make_hook(l):
        def hook(module, inputs, output):
            x = output.reshape(-1, output.shape[-1]).double()
            s = stats[l]
            s[0] += x.shape[0]
            s[1] = s[1] + x.sum(0)
            s[2] = s[2] + x.T @ x
        return hook

    for l in layers:
        expert.layers[l].mlp.register_forward_hook(make_hook(l))
    successes, steps, inputs = [], [], []
    for start in range(0, args.episodes, args.batch_size):
        ids = range(start, start + args.batch_size)
        res = rollout.run_batch(policy, env, procs, ids, args.seed, obs_every=args.obs_every)
        successes += res["success"].tolist()
        steps += res["steps"].tolist()
        inputs += res["inputs"]

    result = {}
    for l, (n, sx, sxx) in stats.items():
        mean = sx / n
        eigvals, eigvecs = torch.linalg.eigh(sxx / n - torch.outer(mean, mean))
        eigvals, eigvecs = eigvals.flip(0).clamp(min=0), eigvecs.flip(1)
        result[l] = {"mean": mean.float().cpu(), "eigvals": eigvals.float().cpu(),
                     "eigvecs": eigvecs.float().cpu(), "n": n}
        ratio = (eigvals.cumsum(0) / eigvals.sum()).cpu().numpy()
        print(f"layer {l}: n={n}, explained variance top1/8/32 = {ratio[0]:.3f}/{ratio[7]:.3f}/{ratio[31]:.3f}")
    torch.save({"layers": result, "success": successes, "steps": steps}, out / "pca.pt")
    keys = [k for k, v in inputs[0].items() if torch.is_tensor(v)]
    bank = {k: torch.stack([x[k] for x in inputs]) for k in keys}
    bank = {k: v.half() if v.is_floating_point() and v.ndim == 4 else v for k, v in bank.items()}  # images
    bank["task"] = [x["task"] for x in inputs]
    torch.save(bank, out / "inputs.pt")
    print(f"{len(inputs)} policy inputs saved for openloop.py")
    print(f"success {sum(successes)}/{len(successes)}; saved {out / 'pca.pt'}")
    env.close()


if __name__ == "__main__":
    main()
