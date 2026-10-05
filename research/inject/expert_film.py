"""FiLM on SmolVLA action-expert MLP outputs: mlp' = (1+γ)⊙mlp + β, [γ;β] = α·W_ℓ z.

The MLP output is the layer's residual update, so β shifts the residual stream after layer ℓ.
Hooks fire only in denoising (prefix prefill skips the expert); all steps share γ, β.
"""
import torch


def attach(policy, layers, dim, alpha, w="random", seed=0):
    """w: "random" (fixed Gaussian / √dim, γ and β) or a fit_pca.py file (β = Σ z_i √λ_i u_i, γ = 0)."""
    expert = policy.model.vlm_with_expert.lm_expert
    H = expert.config.hidden_size
    device = next(expert.parameters()).device
    if w == "random":
        g = torch.Generator().manual_seed(seed)
        W = {l: torch.randn(2 * H, dim, generator=g) / dim**0.5 for l in layers}
    else:
        pca = torch.load(w)["layers"]
        W = {l: torch.cat([torch.zeros(H, dim), pca[l]["eigvecs"][:, :dim] * pca[l]["eigvals"][:dim].sqrt()])
             for l in layers}
    W = {l: (alpha * m).to(device) for l, m in W.items()}
    state = {"z": None}

    def make_hook(l):
        def hook(module, inputs, out):
            z = state["z"]
            assert z is not None and z.shape[0] == out.shape[0], "set_z with one row per env first"
            gb = (z @ W[l].T)[:, None]  # [B, 1, 2H], broadcast over action tokens
            return (out.float() * (1 + gb[..., :H]) + gb[..., H:]).to(out.dtype)
        return hook

    for l in layers:
        expert.layers[l].mlp.register_forward_hook(make_hook(l))

    def set_z(z):
        state["z"] = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(-1, dim)

    return dim, set_z
