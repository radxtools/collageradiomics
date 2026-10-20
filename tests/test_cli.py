import csv

import numpy as np
import pytest
import SimpleITK as sitk
from _collageradiomics.cli import run
from _collageradiomics.io import read_arrays
from _collageradiomics.statistics import summarize
from click.testing import CliRunner


@pytest.fixture
def cases(tmp_path):
    result = {}
    for name, shape in [("two", (7, 9)), ("three", (5, 7, 9)), ("single", (1, 7, 9))]:
        array = np.random.default_rng(51).random(shape)
        mask = np.ones(shape, dtype=np.uint8)
        mask[..., :2] = 0
        paths = []
        for suffix, values in [("image", array), ("mask", mask)]:
            path = tmp_path / f"{name}-{suffix}.mha"
            sitk.WriteImage(sitk.GetImageFromArray(values), str(path))
            paths.append(path)
        result[name] = paths
    return result


def invoke(arguments):
    return CliRunner().invoke(run, [str(arg) for arg in arguments])


def test_sitk_axes_match_height_width_depth(cases):
    image, mask = read_arrays(*cases["three"])
    assert image.shape == mask.shape == (7, 9, 5)
    original = sitk.GetArrayFromImage(sitk.ReadImage(str(cases["three"][0])))
    assert image[2, 3, 4] == original[4, 2, 3]


@pytest.mark.parametrize("name", ["two", "three"])
@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("explicit_dimensions", [False, True])
def test_single_and_batch_csv_match(cases, tmp_path, name, extended, explicit_dimensions):
    image, mask = cases[name]
    single, batch, inputs = (tmp_path / name for name in ["single.csv", "batch.csv", "input.csv"])
    options = ["-s", "1", "-b", "8", "-h", "3"]
    if extended:
        options.append("-e")
    if explicit_dimensions:
        options += ["-d", "2" if name == "two" else "3"]
    first = invoke(["-i", image, "-m", mask, "-o", single, *options])
    assert first.exit_code == 0, first.output
    with inputs.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Image", "Mask"])
        writer.writerow(["case", image.name, mask.name])
    second = invoke(["-i", inputs, "-o", batch, *options])
    assert second.exit_code == 0, second.output
    with single.open() as stream:
        expected = {row["FeatureName"]: float(row["Value"]) for row in csv.DictReader(stream)}
    with batch.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["ID"] == "case"
    assert rows[0]["haralickwindow"] == "3"
    assert {key: float(rows[0][key]) for key in expected} == pytest.approx(expected, abs=1e-8)
    assert len(expected) == 13 * (1 if name == "two" else 2) * (6 if extended else 4)


def test_explicit_2d_single_slice(cases, tmp_path):
    result = invoke(
        [
            "-i",
            cases["single"][0],
            "-m",
            cases["single"][1],
            "-o",
            tmp_path / "out.csv",
            "-d",
            "2",
            "-s",
            "1",
            "-b",
            "8",
        ]
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("property_name,values", [("Origin", (1.0, 0.0)), ("Spacing", (2.0, 1.0))])
def test_misaligned_masks_rejected(cases, property_name, values):
    image, path = cases["two"]
    mask = sitk.ReadImage(str(path))
    getattr(mask, f"Set{property_name}")(values)
    sitk.WriteImage(mask, str(path))
    with pytest.raises(ValueError, match=property_name.lower()):
        read_arrays(image, path)


def test_mask_required(cases, tmp_path):
    result = invoke(["-i", cases["two"][0], "-o", tmp_path / "out.csv"])
    assert result.exit_code == 2
    assert "--mask is required" in result.output


def test_multi_slice_volume_is_not_silently_truncated(cases):
    with pytest.raises(ValueError, match="single-slice"):
        read_arrays(*cases["three"], dimensions=2)


def test_batch_preserves_good_cases_and_reports_failures(cases, tmp_path):
    inputs, output = tmp_path / "input.csv", tmp_path / "out.csv"
    with inputs.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Image", "Mask"])
        writer.writerow(["good", *cases["two"]])
        writer.writerow(["bad", "missing.mha", "missing-mask.mha"])
    result = invoke(["-i", inputs, "-o", output, "-s", "1", "-b", "8"])
    assert result.exit_code == 1
    with output.open() as stream:
        assert [row["ID"] for row in csv.DictReader(stream)] == ["good"]
    with (tmp_path / "errors_out.csv").open() as stream:
        assert [row["ID"] for row in csv.DictReader(stream)] == ["bad"]


def test_window_option_changes_results(cases, tmp_path):
    paths = []
    for size in [3, 5]:
        output = tmp_path / f"window-{size}.csv"
        result = invoke(
            [
                "-i",
                cases["two"][0],
                "-m",
                cases["two"][1],
                "-o",
                output,
                "-s",
                "1",
                "-b",
                "8",
                "-h",
                size,
            ]
        )
        assert result.exit_code == 0, result.output
        paths.append(output.read_text())
    assert paths[0] != paths[1]


def test_label_selection(cases):
    path = cases["two"][1]
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    mask[3:5, 4:6] = 7
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(path))
    _, selected = read_arrays(*cases["two"], label=7)
    assert np.count_nonzero(selected) == 4


def test_constant_and_single_voxel_statistics_are_finite():
    assert summarize(np.array([np.nan, 2.0]), True) == {
        "Median": 2,
        "IQR": 0,
        "Skewness": 0,
        "Kurtosis": 0,
        "Mean": 2,
        "Variance": 0,
    }
