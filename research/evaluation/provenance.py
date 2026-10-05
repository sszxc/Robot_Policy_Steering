"""Record commit and run arguments in a fresh output directory before evaluation."""
import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from platforms.paths import repo_root


def capture(args):
    root = repo_root()
    git = lambda *cmd: subprocess.check_output(["git", *cmd], cwd=root, text=True).strip()
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config": {
            "suite": args.suite, "policy": args.policy, "episodes_per_task": args.episodes,
            "task_ids": json.loads(args.task_ids) if args.task_ids else None,
            "seed": args.seed, "batch_size": args.batch_size,
            "episode_length": args.episode_length or None,
            "init_state_offset": args.init_state_offset,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-length", default="")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--init-state-offset", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "provenance.json").write_text(json.dumps(capture(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
