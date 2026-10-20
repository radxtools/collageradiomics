import numpy as np
import pytest


@pytest.fixture
def image_and_mask_2d():
    image = np.random.default_rng(42).random((50, 52))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[20:26, 21:29] = 1
    mask[22:24, 24:26] = 0
    return image, mask


@pytest.fixture
def image_and_mask_3d():
    image = np.random.default_rng(43).random((50, 52, 7))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[20:26, 21:27, 2:5] = 1
    return image, mask
