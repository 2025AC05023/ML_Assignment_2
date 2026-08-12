"""Utility functions for filesystem and object persistence."""

from pathlib import Path
from typing import Any, Iterable

import joblib


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_joblib(obj: Any, file_path: Path, compress: int | bool = 0) -> None:
    """Persist an object to disk using joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path, compress=compress)


def load_joblib(file_path: Path) -> Any:
    """Load a joblib artifact from disk."""
    return joblib.load(file_path)
