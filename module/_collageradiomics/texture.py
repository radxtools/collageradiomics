"""Map co-occurrence statistics to CoLlAGe's thirteen public feature names."""

from typing import Any

import mahotas
import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def maximal_correlation(matrix: NDArray[Any]) -> float:
    """Second singular value of P(i,j)/sqrt(Px(i)*Py(j)).

    This is the square root of the second eigenvalue of Haralick's Q matrix.
    Empty marginals are removed; a constant variable has coefficient zero.
    See https://haralick.org/journals/dependency_prl_1.pdf, section 3.
    """
    probability = matrix.astype(float) / float(matrix.sum())
    rows, columns = probability.sum(axis=1), probability.sum(axis=0)
    row_indices, column_indices = rows > 0, columns > 0
    if min(np.count_nonzero(row_indices), np.count_nonzero(column_indices)) < 2:
        return 0.0
    normalized = probability[np.ix_(row_indices, column_indices)] / np.sqrt(
        rows[row_indices, None] * columns[None, column_indices]
    )
    singular_values = linalg.svdvals(normalized, check_finite=False)
    return float(np.clip(singular_values[1], 0.0, 1.0))


def named_haralick(matrix: NDArray[Any], x_minus_y_variance: bool) -> NDArray[np.float64]:
    raw = mahotas.features.texture.haralick_features(
        [matrix],
        return_mean=True,
        use_x_minus_y_variance=x_minus_y_variance,
    )
    # Mahotas includes inverse difference moment at index 4. CoLlAGe's public
    # list omits it and includes maximal correlation as the thirteenth feature.
    output = np.empty(13, dtype=float)
    output[:4] = raw[:4]
    output[4:12] = raw[5:13]
    output[12] = maximal_correlation(matrix)
    return output
