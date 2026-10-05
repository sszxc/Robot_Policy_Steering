"""Open-loop sensitivity of the action chunk to z on saved policy inputs (fit_pca.py inputs.pt).

No simulation. Per input with fixed flow noise: Δa for z = c·u_k (dose curves), central-difference
Jacobian ∂a/∂z at |δ| = --fd, and the action spread from resampling noise at z=0. Normalized units.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from research import rollout
from research.inject import INJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    rollout.add_args(parser)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inject", default="expert_film", choices=INJECT)
    parser.add_argument("--layers", default="7,15,23,31")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--w", default="random")
    parser.add_argument("--w-seed", type=int, default=0)
    parser.add_argument("--n-z", type=int, default=8)
    parser.add_argument("--z-seed", type=int, default=0)
    parser.add_argument("--scales", default="0.03,0.1,0.3,1,3")
    parser.add_argument("--fd", type=float, default=0.3, help="central-difference step on each z_i")
    parser.add_argument("--noise-reps", type=int, default=8)
    parser.add_argument("--max-inputs", type=int, default=256)
    args = parser.parse_args()
    out = rollout.new_run_dir(args, f"openloop-{args.suite}-t{args.task_id}")
    policy, cfg = rollout.load_policy(args, rollout.env_config(args))
    dim, set_z = INJECT[args.inject](policy, layers=[int(l) for l in args.layers.split(",")],
                                     dim=args.dim, alpha=args.alpha, w=args.w, seed=args.w_seed)
    bank = torch.load(args.inputs)
    N = min(args.max_inputs, len(bank["task"]))
    idx = np.random.default_rng(0).choice(len(bank["task"]), N, replace=False)
    U = torch.tensor(np.random.default_rng(args.z_seed).standard_normal((args.n_z, dim)), dtype=torch.float32)
    scales = [float(c) for c in args.scales.split(",")]
    A = 7  # real action dims (the rest is padding to max_action_dim)
    B = args.batch_size

    def predict(batch, z, noise):
        set_z(z)
        with torch.inference_mode():
            return policy.predict_action_chunk(dict(batch), noise=noise.clone())[..., :A].float().cpu()

    res = {k: [] for k in ("base", "noise_a", "dose", "jac")}
    for s in range(0, N, B):
        ii = idx[s:s + B]
        b = len(ii)
        batch = {k: (v[ii].cuda().float() if v.is_floating_point() else v[ii].cuda())
                 for k, v in bank.items() if torch.is_tensor(v)}
        batch["task"] = [bank["task"][i] for i in ii]
        shape = (b, cfg.chunk_size, cfg.max_action_dim)
        gen = torch.Generator("cuda").manual_seed(args.seed + s)
        noise = torch.randn(shape, device="cuda", generator=gen)
        zero = torch.zeros(b, dim)
        base = predict(batch, zero, noise)
        res["base"].append(base)
        res["noise_a"].append(torch.stack([predict(batch, zero, torch.randn(shape, device="cuda", generator=gen))
                                           for _ in range(args.noise_reps)], 1))
        res["dose"].append(torch.stack([torch.stack([predict(batch, (c * u).expand(b, dim), noise) - base
                                                     for c in scales], 1) for u in U], 1))
        jac = []
        for i in range(dim):
            e = torch.zeros(b, dim)
            e[:, i] = args.fd
            jac.append((predict(batch, e, noise) - predict(batch, -e, noise)) / (2 * args.fd))
        res["jac"].append(torch.stack(jac, -1))
        print(f"inputs {s + b}/{N}", flush=True)
    res = {k: torch.cat(v).numpy() for k, v in res.items()}
    # base [N, T, A]; noise_a [N, R, T, A]; dose [N, K, S, T, A]; jac [N, T, A, dim]
    np.savez_compressed(out / "openloop.npz", scales=np.array(scales), U=U.numpy(), **res)
    a0 = lambda x: x[..., 0, :]
    noise_sd = a0(res["noise_a"]).std(1).mean()
    for j, c in enumerate(scales):
        d = np.linalg.norm(a0(res["dose"][:, :, j]), axis=-1).mean() / np.sqrt(A)
        print(f"scale {c}: RMS Δa0 {d:.4f} (noise sd {noise_sd:.4f})")
    print(f"saved {out / 'openloop.npz'}")


if __name__ == "__main__":
    main()
