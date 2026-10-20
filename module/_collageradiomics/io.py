"""Read scalar medical images using the same axis and geometry rules."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def read_arrays(
    image_path: str | Path,
    mask_path: str | Path,
    dimensions: int | None = None,
    label: int = -1,
) -> tuple[NDArray[Any], NDArray[Any]]:
    import SimpleITK as sitk

    image, mask = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(mask_path))
    if any(item.GetNumberOfComponentsPerPixel() != 1 for item in (image, mask)):
        raise ValueError("image and mask must be scalar images")
    if image.GetDimension() not in (2, 3):
        raise ValueError("image must be 2D or 3D")
    if image.GetDimension() != mask.GetDimension() or image.GetSize() != mask.GetSize():
        raise ValueError("image and mask must have the same dimensions and size")
    for name in ("Spacing", "Origin", "Direction"):
        left, right = getattr(image, f"Get{name}")(), getattr(mask, f"Get{name}")()
        if not np.allclose(left, right, rtol=0, atol=1e-6):
            raise ValueError(
                f"image and mask {name.lower()} differ; align the mask before extraction"
            )
    array, selection = sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(mask)
    if label != -1:
        selection = selection == label
    requested = image.GetDimension() if dimensions is None else dimensions
    if requested == 2 and array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("--dimensions 2 requires a 2D image or a single-slice volume")
        array, selection = array[0], selection[0]
    elif requested == 3 and array.ndim != 3:
        raise ValueError("--dimensions 3 requires a 3D image")
    if array.ndim == 3:
        # SimpleITK is (z,y,x); the core expects (height,width,depth) = (y,x,z).
        array, selection = np.moveaxis(array, 0, -1), np.moveaxis(selection, 0, -1)
    return array, selection
