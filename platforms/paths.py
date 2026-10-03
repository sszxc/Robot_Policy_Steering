"""Repository resources and machine-specific paths for editable checkouts."""
import json
import os
from pathlib import Path


def repo_root():
    return Path(os.environ.get("THESIS_REPO_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()


def platform_config():
    platform = os.environ.get("THESIS_PLATFORM", "local")
    if platform not in ("local", "hpc"):
        raise ValueError(f"Unknown THESIS_PLATFORM: {platform}")
    directory = repo_root() / "platforms" / platform
    config = json.loads((directory / "config.json").read_text())
    override = Path(os.environ.get("THESIS_PLATFORM_CONFIG", directory / "config.local.json")).expanduser()
    if override.is_file():
        config.update(json.loads(override.read_text()))
    elif "THESIS_PLATFORM_CONFIG" in os.environ:
        raise FileNotFoundError(override)
    return config


def _configured_path(key, env):
    path = Path(os.environ.get(env, platform_config()[key])).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()


def asset_path(*parts):
    return repo_root().joinpath("assets", *parts)


def data_root():
    return _configured_path("data_root", "THESIS_DATA_ROOT")


def output_path(*parts):
    return _configured_path("output_root", "THESIS_OUTPUT_ROOT").joinpath(*parts)
