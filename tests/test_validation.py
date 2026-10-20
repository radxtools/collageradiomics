import numpy as np
import pytest

from collageradiomics import Collage


@pytest.mark.parametrize('name', ['image', 'mask'])
def test_arrays_required(image_and_mask_2d, name):
    image, mask = image_and_mask_2d
    with pytest.raises(TypeError, match=f'{name} must be a NumPy array'):
        Collage(image.tolist() if name == 'image' else image,
                mask.tolist() if name == 'mask' else mask)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('name', ['image', 'mask'])
def test_nonfinite_arrays_rejected(image_and_mask_2d, name, bad_value):
    image, mask = (a.astype(float) for a in image_and_mask_2d)
    (image if name == 'image' else mask)[0, 0] = bad_value
    with pytest.raises(ValueError, match=f'{name} must contain only finite'):
        Collage(image, mask)


@pytest.mark.parametrize('dtype', [object, complex, str])
def test_nonreal_image_rejected(image_and_mask_2d, dtype):
    image, mask = image_and_mask_2d
    with pytest.raises(TypeError, match='image must contain real numeric'):
        Collage(image.astype(dtype), mask)


def test_empty_mask_has_clear_error(image_and_mask_2d):
    image, mask = image_and_mask_2d
    with pytest.raises(ValueError, match='mask must contain at least one nonzero'):
        Collage(image, np.zeros_like(mask))


@pytest.mark.parametrize('name', ['svd_radius', 'haralick_window_size', 'num_unique_angles'])
@pytest.mark.parametrize('value', [True, 1.5, '3'])
def test_integer_parameters_required(image_and_mask_2d, name, value):
    with pytest.raises(TypeError, match='must be an integer'):
        Collage(*image_and_mask_2d, **{name: value})


def test_even_window_rejected(image_and_mask_2d):
    with pytest.raises(ValueError, match='must be odd'):
        Collage(*image_and_mask_2d, haralick_window_size=4)


@pytest.mark.parametrize('values', [[], [np.nan], [[0]], [np.inf]])
def test_invalid_angles_rejected(image_and_mask_2d, values):
    with pytest.raises(ValueError, match='cooccurence_angles'):
        Collage(*image_and_mask_2d, cooccurence_angles=values)


def test_features_require_execute(image_and_mask_2d):
    with pytest.raises(RuntimeError, match=r'execute\(\)'):
        Collage(*image_and_mask_2d).get_single_feature_output(0)


@pytest.mark.parametrize('feature', [-1, 13])
def test_feature_index_range(image_and_mask_2d, feature):
    collage = Collage(*image_and_mask_2d, svd_radius=1, num_unique_angles=8)
    collage.execute()
    with pytest.raises(ValueError, match='which_feature'):
        collage.get_single_feature_output(feature)
