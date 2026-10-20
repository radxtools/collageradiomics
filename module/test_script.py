"""Create numeric feature maps and derived DICOM previews from the bundled sample."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pydicom
from collageradiomics import Collage, HaralickFeature
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from skimage.exposure import equalize_hist


def run_example(input_path: Path, output_dir: Path) -> list[Path]:
    instance = pydicom.dcmread(input_path)
    corrected = apply_voi_lut(apply_modality_lut(instance.pixel_array, instance), instance)
    image = equalize_hist(corrected)
    mask = np.zeros(image.shape, dtype=bool)
    cy, cx = np.asarray(image.shape) // 2
    mask[max(0, cy - 6) : cy + 6, max(0, cx - 6) : cx + 6] = True
    output = Collage(image, mask, svd_radius=2, num_unique_angles=16).execute()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "collage_features.npy", output)
    files = []
    for feature in HaralickFeature:
        values = output[:, :, feature]
        finite = np.isfinite(values)
        low, high = float(values[finite].min()), float(values[finite].max())
        # Reserve stored value zero for padding. Rescaling recovers feature units.
        slope = (high - low) / 65534 if high > low else 1.0
        intercept = low - slope
        pixels = np.zeros(values.shape, dtype="<u2")
        pixels[finite] = np.clip(np.rint((values[finite] - intercept) / slope), 1, 65535).astype(
            "<u2"
        )
        derived = copy.deepcopy(instance)
        derived.SOPClassUID = SecondaryCaptureImageStorage
        derived.SOPInstanceUID = generate_uid()
        derived.SeriesInstanceUID = generate_uid()
        derived.file_meta.MediaStorageSOPClassUID = derived.SOPClassUID
        derived.file_meta.MediaStorageSOPInstanceUID = derived.SOPInstanceUID
        derived.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        derived.ImageType = ["DERIVED", "SECONDARY"]
        derived.SeriesDescription = f"CoLlAGe {feature.name}"
        derived.PhotometricInterpretation = "MONOCHROME2"
        derived.SamplesPerPixel = 1
        derived.BitsAllocated = derived.BitsStored = 16
        derived.HighBit, derived.PixelRepresentation = 15, 0
        derived.PixelPaddingValue = 0
        derived.RescaleSlope, derived.RescaleIntercept = slope, intercept
        for key in ("VOILUTSequence", "ModalityLUTSequence", "RealWorldValueMappingSequence"):
            if key in derived:
                del derived[key]
        derived.WindowCenter = (low + high) / 2
        derived.WindowWidth = max(high - low, 1.0)
        derived.PixelData = pixels.tobytes()
        filename = output_dir / f"collage_{int(feature):02d}_{feature.name}.dcm"
        derived.save_as(filename, enforce_file_format=True)
        files.append(filename)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("test.dcm"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_data"))
    args = parser.parse_args()
    for path in run_example(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
