from platforms.paths import asset_path, data_root, output_path, repo_root, ur5_ip


def test_paths_work_outside_checkout_and_accept_env_overrides(tmp_path, monkeypatch):
    root = repo_root()
    monkeypatch.chdir(tmp_path)
    for name in ("THESIS_OUTPUT_ROOT", "THESIS_DATA_ROOT", "THESIS_UR5_IP"):
        monkeypatch.delenv(name, raising=False)
    assert asset_path("mujoco", "hx5_right.xml").is_file()
    assert data_root() == root / "data"
    assert output_path("recording.npz") == root / "out/recording.npz"
    monkeypatch.setenv("THESIS_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("THESIS_UR5_IP", "127.0.0.1")
    assert output_path() == tmp_path / "outputs"
    assert ur5_ip() == "127.0.0.1"
