"""Consistent named summary statistics for feature maps."""

from typing import Any

import numpy as np
from collageradiomics import HaralickFeature
from numpy.typing import NDArray


def summarize(values: NDArray[Any], extended: bool = False) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("feature contains no finite values")
    centered = finite - np.mean(finite)
    scale = float(np.max(np.abs(centered)))
    # Normalize before calculating moments to avoid cancellation warnings and
    # underflow for nearly constant maps. These are population (biased) moments.
    skewness, kurtosis = 0.0, 0.0
    if scale > 0:
        normalized = centered / scale
        second = float(np.mean(normalized**2))
        skewness = float(np.mean(normalized**3)) / second**1.5
        kurtosis = float(np.mean(normalized**4)) / second**2 - 3
    result = {"Median": float(np.median(finite))}
    if extended:
        result["IQR"] = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    result["Skewness"] = skewness
    result["Kurtosis"] = kurtosis
    if extended:
        result["Mean"] = float(np.mean(finite))
    result["Variance"] = float(np.var(finite, ddof=1)) if finite.size > 1 else 0.0
    return result


def feature_statistics(output: NDArray[Any], extended: bool = False) -> dict[str, float]:
    if output.ndim not in (3, 5):
        raise ValueError("expected a 2D or 3D CoLlAGe feature map")
    result: dict[str, float] = {}
    for feature in HaralickFeature:
        for angle, suffix in [(0, "")] if output.ndim == 3 else [(0, "Theta"), (1, "Phi")]:
            values = output[:, :, feature] if output.ndim == 3 else output[:, :, :, feature, angle]
            for statistic, value in summarize(values, extended).items():
                result[f"Collage{feature.name}{statistic}{suffix}"] = value
    return result
