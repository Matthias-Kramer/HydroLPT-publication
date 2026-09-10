"""Binary cache helpers for reusable binary-map plot payloads."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_binary_map_cache(path: str | Path, payload: dict[str, Any]) -> Path:
    """Serialize a binary-map cache payload to disk."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache_path


def load_binary_map_cache(path: str | Path) -> dict[str, Any]:
    """Load a previously serialized binary-map cache payload."""
    cache_path = Path(path)
    with cache_path.open("rb") as f:
        return pickle.load(f)
