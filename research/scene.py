"""Start positions of the task's objects of interest per init state (LIBERO obj_of_interest:
target object(s), then the goal container). World frame, same as the eef xyz in research.run states.
"""
import argparse
import json

from lerobot.envs.factory import make_env
from research import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    rollout.add_args(parser)
    parser.add_argument("--inits", type=int, default=50)
    parser.add_argument("--out", required=True, help="scene.json path")
    args = parser.parse_args()
    env = make_env(rollout.env_config(args), n_envs=1, use_async_envs=False)[args.suite][args.task_id]
    if hasattr(env, "_ensure"):
        env._ensure()
        env = env._env
    e = env.envs[0].unwrapped
    out = {"task": e.task_description, "objects": None, "inits": {}}
    for r in range(args.inits):
        e.init_state_id = r
        e.reset(seed=args.seed + r)
        problem = e._env.env
        obs = problem._get_observations()
        out["objects"] = list(problem.obj_of_interest)
        out["inits"][r] = {name: obs[f"{name}_pos"].round(4).tolist() for name in out["objects"]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{out['task']}: {out['objects']} for {args.inits} inits -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
