import numpy as np
import pytest
from collageradiomics import Collage


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (7, 9), (5, 6, 1), (5, 6, 2), (7, 9, 4)])
def test_full_masks_and_small_images(shape):
    image = np.ones(shape, dtype=np.uint16)
    mask = np.ones(shape, dtype=bool)
    output = Collage(image, mask, svd_radius=1, num_unique_angles=8).execute()
    assert output.shape == shape + ((13,) if len(shape) == 2 else (13, 2))
    assert np.isfinite(output).all()


@pytest.mark.parametrize("corner", [(0, 0), (0, -1), (-1, 0), (-1, -1)])
def test_single_voxel_corner_masks(corner):
    image = np.random.default_rng(44).random((7, 9))
    mask = np.zeros_like(image, dtype=bool)
    mask[corner] = True
    output = Collage(image, mask, svd_radius=2, num_unique_angles=8).execute()
    assert np.isfinite(output[mask]).all()
    assert np.isnan(output[~mask]).all()


@pytest.mark.parametrize("slices", [slice(0, 1), slice(0, 2), slice(-2, None), slice(-1, None)])
def test_3d_masks_at_volume_ends(slices):
    image = np.random.default_rng(45).random((7, 9, 5))
    mask = np.zeros_like(image, dtype=bool)
    mask[2:5, 3:6, slices] = True
    output = Collage(image, mask, svd_radius=1, num_unique_angles=8).execute()
    assert np.isfinite(output[mask]).all()
    assert np.isnan(output[~mask]).all()


def test_strided_input_matches_contiguous_copy():
    image = np.random.default_rng(46).random((12, 14))[::2, ::2]
    mask = np.ones(image.shape, dtype=bool)
    first = Collage(image, mask, svd_radius=1, num_unique_angles=8).execute()
    second = Collage(image.copy(), mask, svd_radius=1, num_unique_angles=8).execute()
    np.testing.assert_array_equal(first, second)
