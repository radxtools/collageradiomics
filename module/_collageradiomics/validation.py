"""Validate inputs before allocating or calculating feature maps."""

from collections.abc import Iterable
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray


def integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    result = int(value)
    if result < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return result


def validate_arrays(image: NDArray[Any], mask: NDArray[Any]) -> None:
    for name, array in [('image', image), ('mask', mask)]:
        if not isinstance(array, np.ndarray):
            raise TypeError(f'{name} must be a NumPy array')
        if array.dtype.kind not in 'biuf':
            raise TypeError(f'{name} must contain real numeric values')
        if not np.isfinite(array).all():
            raise ValueError(f'{name} must contain only finite values')
    if image.ndim not in (2, 3):
        raise ValueError('Expected a 2D or 3D image.')
    if mask.shape != image.shape:
        raise ValueError('Mask must be the same shape as image.')
    if not all(image.shape):
        raise ValueError('image dimensions must not be empty')
    if not np.any(mask):
        raise ValueError('mask must contain at least one nonzero voxel')


def angles(values: Iterable[float] | None) -> tuple[float, ...]:
    if values is None:
        return tuple(float(x * np.pi / 4) for x in range(8))
    try:
        array = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError('cooccurence_angles must be an iterable of real angles') from exc
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError('cooccurence_angles must contain finite angles')
    return tuple(float(value) for value in array)
