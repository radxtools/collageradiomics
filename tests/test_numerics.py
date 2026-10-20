import math

import numpy as np
import pytest
from _collageradiomics.texture import maximal_correlation, named_haralick
from collageradiomics import _svd_dominant_angles


def test_binary_checkerboard_reference_features():
    # P(0,1)=P(1,0)=1/2; the difference is always one and X determines Y.
    matrix = np.array([[0, 1], [1, 0]], dtype=np.uint32)
    expected = [0.5, 1, -1, 0.25, 1, 0, 0, 1, 0, 0, -1, math.sqrt(1 - math.exp(-2)), 1]
    np.testing.assert_allclose(named_haralick(matrix, True), expected, atol=1e-12)
    assert named_haralick(matrix, False)[8] == pytest.approx(0.25)


def test_independent_binary_reference_features():
    # A uniform 2x2 joint distribution has two bits of joint entropy and no dependence.
    matrix = np.ones((2, 2), dtype=np.uint32)
    expected = [0.25, 0.5, 0, 0.25, 1, 0.5, 1.5, 2, 0.25, 1, 0, 0, 0]
    np.testing.assert_allclose(named_haralick(matrix, True), expected, atol=1e-12)


def test_constant_reference_features():
    expected = np.zeros(13)
    expected[0] = expected[2] = 1
    np.testing.assert_allclose(named_haralick(np.ones((1, 1), dtype=np.uint32), True), expected)


def test_maximal_correlation_handles_empty_and_nonsymmetric_marginals():
    matrix = np.array([[1, 1, 0], [0, 0, 2], [0, 0, 0]], dtype=np.uint32)
    assert maximal_correlation(matrix) == pytest.approx(1)
    independent = np.outer([1, 2, 0], [2, 3, 4]).astype(np.uint32)
    assert maximal_correlation(independent) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "gradient", [(0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 3)]
)
@pytest.mark.parametrize("sign", [1, -1])
def test_principal_gradient_orientation(gradient, sign):
    gx, gy, gz = gradient
    fields = [np.full((5, 5, 3), sign * g, dtype=float) for g in gradient]
    angles = _svd_dominant_angles(*fields, 1, is_3d=True)
    np.testing.assert_allclose(angles[..., 0], math.atan2(gy, gx), atol=1e-6)
    np.testing.assert_allclose(angles[..., 1], math.atan2(gz, math.hypot(gx, gy)), atol=1e-6)
