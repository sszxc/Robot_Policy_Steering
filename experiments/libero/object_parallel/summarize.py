"""Merge ten verified Object tasks from a single-GPU concurrent evaluation."""
import argparse
import json
from pathlib import Path


def summarize(run_dir, episodes, elapsed):
    tasks = []
    for task_id in range(10):
        folder = run_dir / f"task-{task_id}"
        verification = json.loads((folder / "verification.json").read_text())
        info = json.loads((folder / "eval_info.json").read_text())
        provenance = json.loads((folder / "provenance.json").read_text())["config"]
        if len(info["per_task"]) != 1:
            raise ValueError(f"Expected exactly one task in {folder}")
        task = info["per_task"][0]
        if task["task_group"] != "libero_object" or task["task_id"] != task_id:
            raise ValueError(f"Wrong suite/task in {folder}")
        if verification["n_episodes"] != episodes or verification["verified_videos"] != episodes:
            raise ValueError(f"Incomplete evaluation in {folder}")
        if task["n_success"] != verification["n_success"]:
            raise ValueError(f"Success count mismatch in {folder}")
        if provenance["suite"] != "libero_object" or provenance["episodes_per_task"] != episodes:
            raise ValueError(f"Incorrect provenance in {folder}")
        tasks.append({
            "task_id": task_id, "task_description": task["task_description"],
            "n_episodes": episodes, "n_success": verification["n_success"],
            "pc_success": verification["pc_success"], "eval_s": verification["eval_s"],
            "verified_videos": verification["verified_videos"], "output_dir": str(folder),
            "seed": provenance["seed"], "batch_size": provenance["batch_size"],
        })
    total = sum(task["n_episodes"] for task in tasks)
    successes = sum(task["n_success"] for task in tasks)
    return {
        "suite": "libero_object", "concurrent_tasks": 10, "allocated_gpus": 1,
        "episodes_per_task": episodes, "tasks": tasks,
        "n_episodes": total, "n_success": successes, "pc_success": 100 * successes / total,
        "verified_videos": sum(task["verified_videos"] for task in tasks),
        "rollout_and_verification_wall_s": elapsed,
        "episodes_per_minute": total * 60 / elapsed if elapsed > 0 else None,
        "resource_samples": str(run_dir / "resources"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--elapsed", type=float, required=True)
    args = parser.parse_args()
    summary = summarize(args.run_dir, args.episodes, args.elapsed)
    with (args.run_dir / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
