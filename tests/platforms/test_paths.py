import json

from platforms.paths import asset_path, data_root, output_path, platform_config, repo_root


def test_paths_work_outside_checkout_and_accept_machine_overrides(tmp_path, monkeypatch):
    root = repo_root()
    override = tmp_path / "machine.json"
    override.write_text(json.dumps({"output_root": "out/custom", "ur5_ip": "127.0.0.1"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THESIS_PLATFORM", "local")
    monkeypatch.setenv("THESIS_PLATFORM_CONFIG", str(override))
    monkeypatch.delenv("THESIS_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("THESIS_DATA_ROOT", raising=False)
    assert asset_path("mujoco", "hx5_right.xml").is_file()
    assert data_root() == root / "data"
    assert output_path("recording.npz") == root / "out/custom/recording.npz"
    assert platform_config()["ur5_ip"] == "127.0.0.1"
    monkeypatch.setenv("THESIS_OUTPUT_ROOT", str(tmp_path / "outputs"))
    assert output_path() == tmp_path / "outputs"
