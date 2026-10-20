import numpy as np
import pytest
from collageradiomics import Collage, HaralickFeature


def test_2d_output_preserves_shape_and_mask(image_and_mask_2d):
    image, mask = image_and_mask_2d
    image_before, mask_before = image.copy(), mask.copy()
    collage = Collage(image, mask, svd_radius=1, num_unique_angles=8)

    output = collage.execute()

    assert output.shape == image.shape + (13,)
    assert np.isfinite(output[mask != 0]).all()
    assert np.isnan(output[mask == 0]).all()
    assert collage.collage_output is output
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_execute_is_repeatable(image_and_mask_2d):
    collage = Collage(*image_and_mask_2d, svd_radius=1, num_unique_angles=8)
    first = collage.execute().copy()
    np.testing.assert_array_equal(collage.execute(), first)


@pytest.mark.parametrize("feature", [0, HaralickFeature.Contrast])
def test_single_2d_feature_matches_output(image_and_mask_2d, feature):
    collage = Collage(*image_and_mask_2d, svd_radius=1, num_unique_angles=8)
    output = collage.execute()
    np.testing.assert_array_equal(collage.get_single_feature_output(feature), output[:, :, feature])


def test_all_nonzero_mask_labels_are_included(image_and_mask_2d):
    image, mask = image_and_mask_2d
    labels = mask.astype(np.int16)
    labels[20:22, 21:23] = 7
    expected = Collage(image, mask, svd_radius=1, num_unique_angles=8).execute()
    actual = Collage(image, labels, svd_radius=1, num_unique_angles=8).execute()
    np.testing.assert_array_equal(actual, expected)


def test_3d_output_has_both_angles(image_and_mask_3d):
    image, mask = image_and_mask_3d
    collage = Collage(image, mask, svd_radius=1, num_unique_angles=8)
    output = collage.execute()

    assert output.shape == image.shape + (13, 2)
    assert np.isfinite(output[:, :, 3][mask[:, :, 3] != 0]).all()
    assert np.isnan(output[mask == 0]).all()
    np.testing.assert_array_equal(
        collage.get_single_feature_output(HaralickFeature.Contrast),
        output[:, :, :, HaralickFeature.Contrast, :],
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"svd_radius": 0}, "SVD radius"),
        ({"svd_radius": -1, "haralick_window_size": 3}, "SVD radius"),
        ({"haralick_window_size": 0}, "haralick_window_size"),
        ({"num_unique_angles": 0}, "num_unique_angles"),
    ],
)
def test_invalid_parameters_are_rejected(image_and_mask_2d, options, message):
    with pytest.raises(ValueError, match=message):
        Collage(*image_and_mask_2d, **options)


@pytest.mark.parametrize("shape", [(50,), (50, 52, 3, 2)])
def test_invalid_image_dimensions_are_rejected(shape):
    with pytest.raises(ValueError, match="Expected a 2D or 3D image"):
        Collage(np.ones(shape), np.ones(shape))


def test_mismatched_mask_shape_is_rejected(image_and_mask_2d):
    image, _ = image_and_mask_2d
    with pytest.raises(ValueError, match="Mask must be the same shape"):
        Collage(image, np.ones((50, 51)))
