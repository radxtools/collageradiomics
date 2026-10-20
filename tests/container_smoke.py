"""Exercise the installed CLI inside its built container."""

import csv
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

with TemporaryDirectory() as directory:
    root = Path(directory)
    image = np.random.default_rng(70).random((7, 9))
    mask = np.ones(image.shape, dtype=np.uint8)
    for name, values in [("image", image), ("mask", mask)]:
        sitk.WriteImage(sitk.GetImageFromArray(values), str(root / f"{name}.mha"))
    subprocess.run(
        [
            "collageradiomics",
            "-i",
            str(root / "image.mha"),
            "-m",
            str(root / "mask.mha"),
            "-o",
            str(root / "features.csv"),
            "-s",
            "1",
            "-b",
            "8",
        ],
        check=True,
    )
    with (root / "features.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 52
    assert np.isfinite([float(row["Value"]) for row in rows]).all()
