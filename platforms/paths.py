"""Repository resources and machine-specific paths for editable checkouts."""
import os
from pathlib import Path


def repo_root():
    return Path(os.environ.get("THESIS_REPO_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()


def _env_path(env, default):
    path = Path(os.environ.get(env, default)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()


def data_root():
    return _env_path("THESIS_DATA_ROOT", "data")


def output_path(*parts):
    return _env_path("THESIS_OUTPUT_ROOT", "out").joinpath(*parts)
