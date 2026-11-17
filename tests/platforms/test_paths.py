from platforms.paths import data_root, output_path, repo_root


def test_paths_work_outside_checkout_and_accept_env_overrides(tmp_path, monkeypatch):
    root = repo_root()
    monkeypatch.chdir(tmp_path)
    for name in ("RPS_OUTPUT_ROOT", "RPS_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    assert (root / "research").is_dir()
    assert data_root() == root / "data"
    assert output_path("run.json") == root / "out/run.json"
    monkeypatch.setenv("RPS_OUTPUT_ROOT", str(tmp_path / "outputs"))
    assert output_path() == tmp_path / "outputs"
