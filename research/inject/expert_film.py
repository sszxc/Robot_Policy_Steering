"""FiLM on SmolVLA action-expert MLP outputs: mlp' = (1+γ)⊙mlp + β, [γ;β] = α·W_ℓ z.

The MLP output is the layer's residual update, so β shifts the residual stream after layer ℓ.
Hooks fire only in denoising (prefix prefill skips the expert); all steps share γ, β.
"""
import torch


def attach(policy, layers, dim, alpha, w="random", seed=0, center=None):
    """w: "random" (coupled: Gaussian / √dim, every z_i drives γ and β), "split" (z[:dim/2] -> γ, z[dim/2:] -> β,
    each block / √its width, so per-entry scale matches "random") or a fit_pca.py file (β = Σ z_i √λ_i u_i, γ = 0).
    center: γ acts on x̂ = (x − μ)/σ instead of x, i.e. mlp' = mlp + γ⊙x̂ + β. "token" = per sample and channel over
    the action tokens (InstanceNorm); a pool_stats.py file = fixed per-channel mean/var (BatchNorm running stats)."""
    expert = policy.model.vlm_with_expert.lm_expert
    H = expert.config.hidden_size
    device = next(expert.parameters()).device
    g = torch.Generator().manual_seed(seed)
    if w == "random":
        W = {l: torch.randn(2 * H, dim, generator=g) / dim**0.5 for l in layers}
    elif w == "split":
        k = dim // 2
        W = {l: torch.block_diag(torch.randn(H, k, generator=g) / k**0.5,
                                 torch.randn(H, dim - k, generator=g) / (dim - k)**0.5) for l in layers}
    else:
        pca = torch.load(w)["layers"]
        W = {l: torch.cat([torch.zeros(H, dim), pca[l]["eigvecs"][:, :dim] * pca[l]["eigvals"][:dim].sqrt()])
             for l in layers}
    W = {l: (alpha * m).to(device) for l, m in W.items()}
    if center not in (None, "token"):
        S = torch.load(center)["layers"]
        S = {l: (S[l]["mean"].to(device), (S[l]["var"] + 1e-5).sqrt().to(device)) for l in layers}
    state = {"z": None}

    def make_hook(l):
        def hook(module, inputs, out):
            z = state["z"]
            assert z is not None and z.shape[0] == out.shape[0], "set_z with one row per env first"
            gb = (z @ W[l].T)[:, None]  # [B, 1, 2H], broadcast over action tokens
            x = out.float()
            if center is None:
                return (x * (1 + gb[..., :H]) + gb[..., H:]).to(out.dtype)
            if center == "token":
                xh = (x - x.mean(1, keepdim=True)) / (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            else:
                xh = (x - S[l][0]) / S[l][1]
            return (x + gb[..., :H] * xh + gb[..., H:]).to(out.dtype)
        return hook

    for l in layers:
        expert.layers[l].mlp.register_forward_hook(make_hook(l))

    def set_z(z):
        state["z"] = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(-1, dim)

    return dim, set_z
