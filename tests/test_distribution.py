import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

import pydicom
import pytest
from packaging.requirements import Requirement

REPOSITORY = Path(__file__).resolve().parents[1]


def run_command(arguments, cwd, env=None):
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("distribution")
    source = workspace / "source"
    shutil.copytree(
        REPOSITORY / "module",
        source,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", "docs"),
    )
    output = workspace / "dist"
    # The default build creates an sdist, then builds the wheel from that sdist.
    # Use another working directory to catch accidental dependencies on the checkout.
    run_command(
        ["-m", "build", "--no-isolation", "--outdir", str(output), str(source)],
        cwd=workspace,
    )
    return next(output.glob("*.tar.gz")), next(output.glob("*.whl"))


def test_source_distribution_contains_build_inputs_and_sample(distributions):
    source, _ = distributions
    with tarfile.open(source) as archive:
        names = {
            Path(name).relative_to(Path(source.name.removesuffix(".tar.gz")))
            for name in archive.getnames()
        }
    assert {
        Path(name)
        for name in [
            "VERSION",
            "requirements.txt",
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "setup.py",
            "test.dcm",
            "test_script.py",
        ]
    } <= names


def test_packaged_sample_is_a_readable_dicom(distributions):
    source, _ = distributions
    with tarfile.open(source) as archive:
        sample = next(
            member for member in archive.getmembers() if member.name.endswith("/test.dcm")
        )
        with archive.extractfile(sample) as stream:
            dataset = pydicom.dcmread(stream)
    assert dataset.pixel_array.shape == (dataset.Rows, dataset.Columns)


def test_wheel_metadata_matches_declared_version_and_requirements(distributions):
    _, wheel = distributions
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))

    assert metadata["Name"] == "collageradiomics"
    assert metadata["Version"] == (REPOSITORY / "module" / "VERSION").read_text().strip()
    expected = {
        str(Requirement(line))
        for line in (REPOSITORY / "module" / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    runtime = {
        str(Requirement(line))
        for line in metadata.get_all("Requires-Dist")
        if Requirement(line).marker is None
    }
    assert runtime == expected


def test_built_wheel_installs_and_runs_outside_checkout(distributions, tmp_path):
    _, wheel = distributions
    destination = tmp_path / "installed"
    run_command(
        [
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(destination),
            str(wheel),
        ],
        cwd=tmp_path,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(destination)
    run_command(
        [
            "-c",
            """
from pathlib import Path
import numpy as np
import collageradiomics

assert Path(collageradiomics.__file__).resolve().is_relative_to(Path('installed').resolve())
image = np.ones((50, 50))
mask = np.zeros(image.shape, dtype=np.uint8)
mask[20:26, 20:26] = 1
output = collageradiomics.Collage(image, mask, svd_radius=1).execute()
assert output.shape == (50, 50, 13)
assert np.isfinite(output[mask != 0]).all()
assert np.isnan(output[mask == 0]).all()
""",
        ],
        cwd=tmp_path,
        env=env,
    )
