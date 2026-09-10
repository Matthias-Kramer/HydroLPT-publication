from __future__ import annotations

import numpy as np


OUTSIDE_POLICIES = ("stop",)


def apply_outside_stop(
    indices: np.ndarray,
    *,
    t_now: float,
    x_new: np.ndarray,
    y_new: np.ndarray,
    z_new: np.ndarray,
    tid_prev: np.ndarray,
    x_prev: np.ndarray,
    y_prev: np.ndarray,
    z_prev: np.ndarray,
    tid_at_prev: np.ndarray,
    stopped_by_outside: np.ndarray,
    t_stop_outside: np.ndarray,
) -> None:
    """Mark particles as outside-domain terminated and restore prior state."""
    idx = np.asarray(indices, dtype=int).ravel()
    if idx.size == 0:
        return

    stopped_by_outside[idx] = True
    new = np.isnan(t_stop_outside[idx])
    t_stop_outside[idx[new]] = t_now

    x_new[idx] = x_prev[idx]
    y_new[idx] = y_prev[idx]
    z_new[idx] = z_prev[idx]
    tid_prev[idx] = tid_at_prev[idx]


def validate_outside_policy(outside_policy: str) -> None:
    """Validate the outside-domain interaction policy."""
    if outside_policy not in OUTSIDE_POLICIES:
        raise ValueError('outsidePolicy must be "stop".')
