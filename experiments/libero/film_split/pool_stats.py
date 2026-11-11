"""Pooled per-channel mean/var of expert MLP outputs over the scan PCA fits of all tasks except one (expert_film center=).

Usage: pool_stats.py OUT.pt EXCLUDE_TASK. Leaving the task out mimics BatchNorm running stats on an unseen task.
"""
import sys
from pathlib import Path

import torch

out, skip = Path(sys.argv[1]), int(sys.argv[2])
fits = [torch.load(f"out/libero/smolvla/steer/scan/t{t}/pca.pt")["layers"] for t in range(10) if t != skip]
res = {}
for l in fits[0]:
    n = torch.tensor([float(f[l]["n"]) for f in fits])
    mean = torch.stack([f[l]["mean"].double() for f in fits])
    var = torch.stack([(f[l]["eigvecs"].double() ** 2) @ f[l]["eigvals"].double() for f in fits])  # diag(Σ)
    m = (n[:, None] * mean).sum(0) / n.sum()
    v = (n[:, None] * (var + mean**2)).sum(0) / n.sum() - m**2
    res[l] = {"mean": m.float(), "var": v.float()}
    print(f"layer {l}: |μ| {m.norm():.3f}, RMS σ {v.mean().sqrt():.4f}")
torch.save({"layers": res, "exclude": skip}, out)
