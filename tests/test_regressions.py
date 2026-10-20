import numpy as np
import pytest
from collageradiomics import Collage, DifferenceVarianceInterpretation, HaralickFeature


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_masks_touching_image_edges(image_and_mask_2d, edge):
    image, mask = image_and_mask_2d
    mask[:] = 0
    regions = {
        "left": (slice(20, 26), slice(0, 6)),
        "right": (slice(20, 26), slice(-6, None)),
        "top": (slice(0, 6), slice(21, 27)),
        "bottom": (slice(-6, None), slice(21, 27)),
    }
    mask[regions[edge]] = 1
    output = Collage(image, mask, svd_radius=2, num_unique_angles=8).execute()

    assert output.shape == image.shape + (13,)
    assert np.isfinite(output[mask != 0]).all()
    assert np.isnan(output[mask == 0]).all()


@pytest.mark.parametrize("depth", [2, 3])
def test_every_masked_3d_slice_is_calculated(image_and_mask_3d, depth):
    image, mask = image_and_mask_3d
    mask[:] = 0
    mask[20:26, 21:27, 2 : 2 + depth] = 1
    output = Collage(image, mask, svd_radius=1, num_unique_angles=8).execute()
    assert np.isfinite(output[mask != 0]).all()


@pytest.fixture
def checkerboard_features(image_and_mask_2d):
    # Horizontal pairs alternate 0->1 and 1->0 equally, so P(0,1)=P(1,0)=1/2.
    checkerboard = (np.indices((8, 8)).sum(axis=0) % 2).astype(np.uint8)

    def calculate(**options):
        collage = Collage(
            *image_and_mask_2d,
            svd_radius=1,
            haralick_window_size=3,
            num_unique_angles=2,
            cooccurence_angles=[0],
            **options,
        )
        return collage._calculate_haralick_feature_values(checkerboard, 3, 3)

    return calculate


def test_checkerboard_has_known_energy_contrast_and_correlation(checkerboard_features):
    features = checkerboard_features()
    assert features[HaralickFeature.AngularSecondMoment] == pytest.approx(0.5)
    assert features[HaralickFeature.Contrast] == pytest.approx(1.0)
    assert features[HaralickFeature.Correlation] == pytest.approx(-1.0)


def test_checkerboard_joint_entropy_is_one_bit(checkerboard_features):
    assert checkerboard_features()[HaralickFeature.Entropy] == pytest.approx(1.0)


def test_difference_variance_interpretations_produce_different_results(checkerboard_features):
    x_minus_y = checkerboard_features(
        difference_variance_interpretation=DifferenceVarianceInterpretation.XMinusYVariance,
    )
    probability = checkerboard_features(
        difference_variance_interpretation=DifferenceVarianceInterpretation.ProbabilityXMinusYVariance,
    )
    assert not np.allclose(x_minus_y, probability)
