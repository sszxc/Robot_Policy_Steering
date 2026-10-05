"""Injection configs: each attach(policy, **args) -> (dim, set_z); set_z takes z[B, dim]."""
from . import expert_film

INJECT = {"expert_film": expert_film.attach}
