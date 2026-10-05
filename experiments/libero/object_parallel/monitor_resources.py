"""Sample GPU metrics and the launcher's process tree without extra packages."""
import argparse
import csv
import io
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def process_tree(root_pid):
    result = subprocess.run(
        ["ps", "-e", "-o", "pid=,ppid=,pcpu=,rss=,nlwp=,comm="],
        check=True, capture_output=True, text=True,
    )
    rows = [line.split(maxsplit=5) for line in result.stdout.splitlines() if line.strip()]
    selected = {root_pid}
    while True:
        children = {int(row[0]) for row in rows if int(row[1]) in selected}
        expanded = selected | children
        if expanded == selected:
            break
        selected = expanded
    return [row for row in rows if int(row[0]) in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # ps %CPU is a lifetime average per process, not instantaneous CPU use.
    # RSS includes shared pages; use Slurm MaxRSS/seff for accounting as well.
    with (args.output_dir / "gpu.csv").open("w", buffering=1) as gpu_file, \
         (args.output_dir / "processes.csv").open("w", buffering=1) as process_file:
        gpu_writer = csv.writer(gpu_file)
        process_writer = csv.writer(process_file)
        gpu_writer.writerow([
            "timestamp_utc", "gpu_uuid", "gpu_util_percent", "memory_util_percent",
            "memory_used_mib", "memory_total_mib", "power_w", "power_limit_w",
        ])
        process_writer.writerow([
            "timestamp_utc", "pid", "ppid", "cpu_lifetime_percent", "rss_kib", "threads", "command",
        ])
        while running:
            timestamp = datetime.now(timezone.utc).isoformat()
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, check=True, timeout=15)
            for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
                gpu_writer.writerow([timestamp, *row])
            rows = process_tree(args.root_pid)
            for row in rows:
                process_writer.writerow([timestamp, *row])
            if not any(int(row[0]) == args.root_pid for row in rows):
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
