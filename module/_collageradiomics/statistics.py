"""Consistent named summary statistics for feature maps."""

from typing import Any

import numpy as np
from collageradiomics import HaralickFeature
from numpy.typing import NDArray
from scipy import stats


def summarize(values: NDArray[Any], extended: bool = False) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("feature contains no finite values")
    constant = bool(np.all(finite == finite[0]))
    result = {"Median": float(np.median(finite))}
    if extended:
        result["IQR"] = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    result["Skewness"] = 0.0 if constant else float(stats.skew(finite))
    result["Kurtosis"] = 0.0 if constant else float(stats.kurtosis(finite))
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
