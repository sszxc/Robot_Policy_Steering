"""Summarize openloop.npz: dose curves (heavy-tailed, see percentiles), Jacobian rank, cross-state consistency."""
import sys

import numpy as np


def summarize(path):
    d = np.load(path)
    J0, Jc = d["jac"][:, 0], d["jac"].reshape(len(d["jac"]), -1, d["jac"].shape[-1])  # [N, A, dim], [N, T·A, dim]
    a0 = lambda x: x[..., 0, :]
    noise_sd = a0(d["noise_a"]).std(1).mean()
    out = {"noise_sd_a0": float(noise_sd), "noise_sd_chunk": float(d["noise_a"].std(1).mean())}
    out["dose_a0"] = {float(c): float(np.sqrt((a0(d["dose"][:, :, j]) ** 2).mean())) for j, c in enumerate(d["scales"])}
    out["dose_chunk"] = {float(c): float(np.sqrt((d["dose"][:, :, j] ** 2).mean())) for j, c in enumerate(d["scales"])}
    for name, J in (("a0", J0), ("chunk", Jc)):
        s = np.linalg.svd(J, compute_uv=False)  # per state
        pr = (s.sum(1) ** 2 / (s ** 2).sum(1)).mean()
        sp = np.linalg.svd(J.reshape(-1, J.shape[-1]) if name == "a0" else J.mean(0), compute_uv=False)
        # 1 = same Jacobian at every state; 1/N = unrelated
        consist = (np.linalg.norm(J.mean(0)) ** 2) / (np.linalg.norm(J, axis=(1, 2)) ** 2).mean()
        out[name] = {"gain_rms": float(np.sqrt((J ** 2).mean())), "eff_rank_per_state": float(pr),
                     "sv_per_state_mean": (s.mean(0) / s.mean(0)[0]).round(3).tolist(),
                     "consistency": float(consist)}
    x = np.linalg.norm(a0(d["dose"]), axis=-1) / np.sqrt(J0.shape[1])  # [N, K, S] per-state RMS Δa0
    out["dose_a0_pct10_50_90_99"] = {float(c): np.percentile(x[..., j], [10, 50, 90, 99]).round(4).tolist()
                                     for j, c in enumerate(d["scales"])}
    return out


if __name__ == "__main__":
    import json
    for p in sys.argv[1:]:
        print(p)
        print(json.dumps(summarize(p), indent=1))
