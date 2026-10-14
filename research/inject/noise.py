"""Fixed initial flow noise (Golden-Ticket style): x0 = U z replaces the sampled noise at every denoising call.

dim = chunk * max_action_dim (1600): U = I. Smaller dim: U = random orthonormal columns * sqrt(N / dim),
so z ~ N(0, I) gives x0 entries ~ N(0, 1). z = 0 is the zero-noise policy, not the base policy.
"""
import torch


def attach(policy, dim, seed=0):
    cfg = policy.config
    shape = (cfg.chunk_size, cfg.max_action_dim)
    N = shape[0] * shape[1]
    device = next(policy.parameters()).device
    if dim == N:
        U = torch.eye(N)
    else:
        q, _ = torch.linalg.qr(torch.randn(N, dim, generator=torch.Generator().manual_seed(seed)))
        U = q * (N / dim) ** 0.5
    U = U.to(device)
    state = {"x0": None}

    def sample_noise(s, dev):
        x0 = state["x0"]
        assert x0 is not None and x0.shape[0] == s[0], "set_z with one row per env first"
        return x0.to(dev)

    policy.model.sample_noise = sample_noise

    def set_z(z):
        z = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(-1, dim)
        state["x0"] = (z @ U.T).reshape(-1, *shape)

    return dim, set_z
