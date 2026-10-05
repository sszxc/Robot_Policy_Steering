# Research Target
Target: discover a compact (low-dim), expressive control space over a pretrained robot policy, such that the policy can be efficiently steered or adapted at test time without modifying its full parameters

Ideally, z is not only style, but also a policy control/adaptation manifold.

z can be optimized by CEM, RL, VLM (ICL), human, etc.

Current simulation will based on SmolVLA pretrained model and LIBERO benchmark.

The final hardware experiment for this project will use a UR5 (not a UR5e) robotic arm and a HX5-D20-MRT-Right multi-fingered dexterous hand.


# Computing Environment
Small validation experiments can be run locally.
Use Python: /home/lab/miniforge3/bin/python


There's a remote supercomputer available for use. Start with `docs/ASU_Sol_HPC_quick.md` for common commands; consult `docs/ASU_Sol_HPC_reference.md` for details and `skills/ASU_Sol_HPC` for task-specific guidance. Prefer current cluster queries over generic skill defaults. My asurite ID is: `xzhan742`.
Repo root: ~/Coding/thesis
Use Python env: /scratch/xzhan742/thesis/envs/smolvla-libero-py312-v3
Data root: /scratch/xzhan742/thesis
Note that remote resources are precious. Please make the most of the resources you've requested, especially the GPU VRAM.

# Code Structure
This project currently has only one user and is in a state of agile development. If we need to refactor or adjust the APIs, we don't have to worry about backwards compatibility.
Layout: `docs/architecture.md`. `lerobot/` is a frozen subtree; relevant parts are SmolVLA (now), Pi0.5 / GR00T (possible later) and the LIBERO env.
