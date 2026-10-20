"""SVD of local gradient patches; all spatial arrays use (height,width,depth)."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy import linalg

Array = NDArray[Any]


def dominant_angle(
    x: int,
    y: int,
    z: int,
    dx_windows: Array,
    dy_windows: Array,
    dz_windows: Array,
    is_3d: bool | None = None,
) -> float | tuple[float, float]:
    is_3d = dx_windows.shape[2] > 1 if is_3d is None else is_3d
    patches = [dx_windows[y, x, z], dy_windows[y, x, z]]
    if is_3d:
        patches.append(dz_windows[y, x, z])
    gradients = np.column_stack([patch.ravel(order="F") for patch in patches])
    _, _, vh = linalg.svd(gradients, full_matrices=False, check_finite=False)
    principal = vh[0].copy()
    if principal[np.argmax(np.abs(principal))] < 0:
        principal *= -1
    gx, gy = float(principal[0]), float(principal[1])
    theta = 0.0 if math.hypot(gx, gy) < np.finfo(float).eps else math.atan2(gy, gx)
    if is_3d:
        return theta, math.atan2(float(principal[2]), math.hypot(gx, gy))
    return theta


def dominant_angles(
    dx: Array,
    dy: Array,
    dz: Array,
    svd_radius: int,
    is_3d: bool | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    is_3d = dx.shape[2] > 1 if is_3d is None else is_3d
    window = (2 * svd_radius + 1, 2 * svd_radius + 1, 1)
    # NumPy supplies this rolling view; no third-party source is copied or vendored.
    windows = [sliding_window_view(field, window) for field in (dx, dy, dz)]
    shape = windows[0].shape[:3]
    output = np.zeros(shape + (2 if is_3d else 1,), dtype=np.float32)
    total = math.prod(shape)
    if progress:
        progress(0, total)
    for completed, (y, x, z) in enumerate(np.ndindex(shape), 1):
        output[y, x, z] = dominant_angle(x, y, z, windows[0], windows[1], windows[2], is_3d)
        if progress:
            progress(completed, total)
    return output
