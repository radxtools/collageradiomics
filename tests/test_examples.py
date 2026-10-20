import importlib.util
from pathlib import Path

import numpy as np
import pydicom

REPOSITORY = Path(__file__).resolve().parents[1]


def load_example(path):
    specification = importlib.util.spec_from_file_location("collage_example", REPOSITORY / path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_dicom_example_writes_valid_derived_instances(tmp_path):
    example = load_example("module/test_script.py")
    files = example.run_example(REPOSITORY / "module/test.dcm", tmp_path)
    assert len(files) == 13
    identifiers = set()
    for path in files:
        dataset = pydicom.dcmread(path)
        pixels = dataset.pixel_array
        assert pixels.shape == (dataset.Rows, dataset.Columns)
        assert dataset.PixelPaddingValue == 0
        assert dataset.SOPInstanceUID == dataset.file_meta.MediaStorageSOPInstanceUID
        assert np.isfinite(
            pixels.astype(float) * dataset.RescaleSlope + dataset.RescaleIntercept
        ).all()
        identifiers.add(dataset.SOPInstanceUID)
    assert len(identifiers) == 13
    maps = np.load(tmp_path / "collage_features.npy")
    assert maps.shape[-1] == 13
    assert np.isfinite(maps).any() and np.isnan(maps).any()


def test_isbi_example_covers_every_window_feature_and_angle():
    example = load_example("examples/isbi.py")
    image = np.random.default_rng(65).random((7, 9, 3))
    mask = np.ones(image.shape, dtype=bool)
    result = example.compute_collage(
        image, mask, haralick_windows=[3, 5], svd_radius=1, num_unique_angles=8
    )
    assert result.shape == (2 * 13 * 2, 4)
    assert np.isfinite(result).all()
    assert np.any(result != 0, axis=1).all()
