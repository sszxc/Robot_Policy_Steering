import json
import os
import subprocess
import sys
from pathlib import Path

from platforms.paths import repo_root


def test_launcher_records_inputs_and_preserves_existing_results(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    executable = commands / "lerobot-eval"
    executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n")
    executable.chmod(0o755)
    output = tmp_path / "run"
    capture = tmp_path / "args.txt"
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
               LIBERO_CONFIG_PATH=str(tmp_path / "libero"), OUTPUT_DIR=str(output),
               TASK_IDS="[0]", EPISODES="1", SEED="1000", EPISODE_LENGTH="12",
               POLICY="fixture/model", CAPTURE_ARGS=str(capture))
    Path(env["LIBERO_CONFIG_PATH"]).mkdir()
    Path(env["LIBERO_CONFIG_PATH"], "config.yaml").touch()
    launcher = repo_root() / "experiments/libero/run_libero10_smolvla.sh"
    subprocess.run(["bash", str(launcher)], cwd=tmp_path, env=env, check=True)
    record = json.loads((output / "provenance.json").read_text())
    assert record["config"]["task_ids"] == [0]
    assert record["config"]["seed"] == 1000
    assert record["config"]["policy"] == "fixture/model"
    args = capture.read_text().splitlines()
    assert "--eval.n_episodes=1" in args
    assert "--env.episode_length=12" in args
    assert "--env.control_mode=relative" in args
    before = (output / "provenance.json").read_bytes()
    result = subprocess.run(["bash", str(launcher)], cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode != 0
    assert (output / "provenance.json").read_bytes() == before


def test_aggregate_reads_old_format_without_overwriting_baseline(tmp_path):
    for task_id in range(10):
        folder = tmp_path / "smolvla/vanilla" / f"libero10-task{task_id}-123_{task_id}"
        folder.mkdir(parents=True)
        overall = dict(n_episodes=10, n_success=task_id, pc_success=10 * task_id,
                       eval_s=1, verified_videos=10)
        (folder / "verification.json").write_text(json.dumps(overall))
        (folder / "eval_info.json").write_text(json.dumps({"per_task": [{
            "task_group": "libero_10", "task_id": task_id,
            "task_description": f"Task {task_id}", "n_success": task_id}]}))
    historical = tmp_path / "smolvla/vanilla/libero10-array-123"
    historical.mkdir()
    (historical / "summary.json").write_text("historical result\n")
    out = tmp_path / "migration-check"
    env = dict(os.environ, THESIS_OUTPUT_ROOT=str(tmp_path))
    subprocess.run([sys.executable, "-m", "research.evaluation.aggregate_eval", "123",
                    "--output-dir", str(out)], env=env, check=True, capture_output=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_success"] == 45
    assert summary["n_episodes"] == summary["verified_videos"] == 100
    assert (historical / "summary.json").read_text() == "historical result\n"
