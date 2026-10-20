"""Extract CoLlAGe from an aligned IBSI image/mask pair without hard-coded dataset paths."""

import argparse
from pathlib import Path

import numpy as np
from _collageradiomics.io import read_arrays
from _collageradiomics.statistics import summarize
from collageradiomics import Collage, HaralickFeature


def compute_collage(
    image,
    mask,
    haralick_windows=(3, 5, 7, 9, 11),
    output_dir=None,
    svd_radius=5,
    num_unique_angles=64,
):
    """Return rows (window,feature,angle), with mean, population SD, skew, kurtosis."""
    angles = 2 if image.ndim == 3 else 1
    features = np.empty((len(haralick_windows) * len(HaralickFeature) * angles, 4))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    for window_index, window in enumerate(haralick_windows):
        output = Collage(
            image,
            mask,
            svd_radius=svd_radius,
            num_unique_angles=num_unique_angles,
            haralick_window_size=window,
        ).execute()
        if output_dir is not None:
            np.save(output_dir / f"COLLAGE_RAW_W{window}.npy", output)
        for feature in HaralickFeature:
            for angle in range(angles):
                values = output[:, :, feature] if angles == 1 else output[:, :, :, feature, angle]
                finite = values[np.isfinite(values)]
                statistics = summarize(finite, extended=True)
                index = (window_index * len(HaralickFeature) + int(feature)) * angles + angle
                features[index] = [
                    statistics["Mean"],
                    float(np.std(finite)),
                    statistics["Skewness"],
                    statistics["Kurtosis"],
                ]
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output_data"))
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 5, 7, 9, 11])
    args = parser.parse_args()
    image, mask = read_arrays(args.image, args.mask)
    values = compute_collage(image, mask, args.windows, output_dir=args.output_dir)
    np.save(args.output_dir / "COLLAGE_STATS.npy", values)


if __name__ == "__main__":
    main()
