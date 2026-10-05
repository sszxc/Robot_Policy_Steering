"""Record evaluation inputs before running a policy, without importing ML stacks."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from platforms.paths import repo_root


def capture(args):
    root = repo_root()
    baseline = root / "experiments/libero/baselines/local-baseline.json"
    packages = {}
    for name in ("torch", "mujoco", "hf-libero", "lerobot", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    policy_config = Path(args.policy) / "config.json"
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version,
        "packages": packages,
        "config": {
            "suite": args.suite, "control_mode": "relative", "batch_size": args.batch_size,
            "policy": args.policy, "episodes_per_task": args.episodes,
            "task_ids": json.loads(args.task_ids) if args.task_ids else None,
            "seed": args.seed, "episode_length": args.episode_length or None,
            "init_state_offset": args.init_state_offset,
            "output_dir": str(args.output_dir.resolve()),
        },
        "reference_baseline": json.loads(baseline.read_text()),
        "reference_baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "policy_config_sha256": hashlib.sha256(policy_config.read_bytes()).hexdigest() if policy_config.is_file() else None,
        "model_paths": json.loads(Path(args.model_paths).read_text()) if args.model_paths else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-length", default="")
    parser.add_argument("--model-paths")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--init-state-offset", type=int, default=0)
    args = parser.parse_args()
    record = capture(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "provenance.json").open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
