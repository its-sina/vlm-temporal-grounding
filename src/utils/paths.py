"""Portable default locations for data, caches and results.

Every location can be overridden on the command line; these defaults are used otherwise.
Environment variables take precedence over the project-relative defaults:

    VTG_DATA_DIR     datasets (annotations and videos)       default: <project>/data
    VTG_CACHE_DIR    decoded-frame caches                     default: <project>/cache
    VTG_RESULTS_DIR  predictions and summaries                default: <project>/results
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_dir(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def data_dir() -> Path:
    return _env_dir("VTG_DATA_DIR", PROJECT_ROOT / "data")


def cache_dir() -> Path:
    return _env_dir("VTG_CACHE_DIR", PROJECT_ROOT / "cache")


def results_dir() -> Path:
    return _env_dir("VTG_RESULTS_DIR", PROJECT_ROOT / "results")


def default_annotation_file(dataset: str, split: str, root: Path | None = None) -> Path:
    """Conventional annotation location: <data_dir>/<dataset>/annotations/<split>.json."""
    return (root or data_dir()) / dataset / "annotations" / f"{split}.json"


def default_video_dir(dataset: str, root: Path | None = None) -> Path:
    """Conventional video location: <data_dir>/<dataset>/videos."""
    return (root or data_dir()) / dataset / "videos"


def frame_cache_dir(dataset: str, fps: float, thumb: int, root: Path | None = None) -> Path:
    """Directory of decoded-frame caches for one sampling configuration."""
    return (root or cache_dir()) / "frames" / f"{dataset}_f{fps}_ts{thumb}"
