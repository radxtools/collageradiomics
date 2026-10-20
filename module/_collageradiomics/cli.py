"""Command-line extraction for one image or an ID,Image,Mask CSV."""

import csv
import logging
from pathlib import Path
from typing import Any

import click
from collageradiomics import Collage, DifferenceVarianceInterpretation

from .io import read_arrays
from .statistics import feature_statistics


def extract(
    image: Path,
    mask: Path,
    dimensions: int | None,
    label: int,
    extended: bool,
    options: dict[str, Any],
) -> dict[str, float]:
    arrays = read_arrays(image, mask, dimensions, label)
    return feature_statistics(Collage(*arrays, **options).execute(), extended)


def write_single(output: Path, values: dict[str, float]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["FeatureName", "Value"])
        writer.writerows((name, f"{value:.10f}") for name, value in values.items())


def write_batch(
    input_path: Path,
    output: Path,
    dimensions: int | None,
    label: int,
    extended: bool,
    options: dict[str, Any],
) -> None:
    errors_path = output.with_name("errors_" + output.name)
    if errors_path.resolve() == input_path.resolve():
        raise ValueError("the errors report would overwrite the input CSV")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    fields = ["ID", "Image", "Mask", "svdradius", "haralickwindow", "binsize", "label"]
    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"ID", "Image", "Mask"} <= set(reader.fieldnames or []):
            raise ValueError("input CSV must have ID, Image, and Mask columns")
        for number, row in enumerate(reader, 2):
            identity = {name: row.get(name) or "" for name in ("ID", "Image", "Mask")}
            try:
                if not all(identity.values()):
                    raise ValueError(f"CSV row {number} requires ID, Image, and Mask")
                paths = [Path(identity[name]).expanduser() for name in ("Image", "Mask")]
                image, mask = [p if p.is_absolute() else input_path.parent / p for p in paths]
                values = extract(image, mask, dimensions, label, extended, options)
                metadata = dict(
                    identity,
                    svdradius=options["svd_radius"],
                    haralickwindow=options["haralick_window_size"],
                    binsize=options["num_unique_angles"],
                    label=label,
                )
                rows.append(dict(metadata, **values))
                fields.extend(key for key in values if key not in fields)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(dict(identity, Error=str(exc)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    with errors_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, ["ID", "Image", "Mask", "Error"])
        writer.writeheader()
        writer.writerows(errors)
    if errors:
        raise ValueError(f"{len(errors)} case(s) failed; see {errors_path}")


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Image or CSV with ID, Image, Mask columns; CSV paths are relative to the CSV.",
)
@click.option(
    "-m",
    "--mask",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Mask image; required for a single image. All nonzero labels are included by default.",
)
@click.option("-o", "--outputfile", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Show extraction progress.")
@click.option(
    "-d", "--dimensions", type=click.IntRange(2, 3), help="Defaults to image dimensionality."
)
@click.option("-s", "--svdradius", default=5, show_default=True, type=click.IntRange(min=1))
@click.option(
    "-h",
    "--haralickwindow",
    default=-1,
    type=int,
    help="Odd window size >=3; -1 uses 2*svdradius+1.",
)
@click.option("-b", "--binsize", default=64, show_default=True, type=click.IntRange(min=2))
@click.option(
    "-l", "--label", default=-1, type=click.IntRange(min=-1), help="-1 includes all nonzero labels."
)
@click.option("-e", "--extendstats", is_flag=True, help="Also include mean and IQR.")
@click.option(
    "--difference-variance", type=click.Choice(["x-minus-y", "probability"]), default="x-minus-y"
)
@click.option(
    "--cooccurrence-angle",
    type=float,
    multiple=True,
    help="GLCM angle in radians; repeat as needed.",
)
def run(
    input_path: Path,
    mask: Path | None,
    outputfile: Path,
    verbose: bool,
    dimensions: int | None,
    svdradius: int,
    haralickwindow: int,
    binsize: int,
    label: int,
    extendstats: bool,
    difference_variance: str,
    cooccurrence_angle: tuple[float, ...],
) -> None:
    """Extract CoLlAGe feature summaries from aligned scalar images and masks."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s: %(message)s"
    )
    if outputfile.resolve() in {input_path.resolve(), mask.resolve() if mask else None}:
        raise click.UsageError("output must differ from the input and mask files")
    options = {
        "svd_radius": svdradius,
        "haralick_window_size": 2 * svdradius + 1 if haralickwindow == -1 else haralickwindow,
        "num_unique_angles": binsize,
        "cooccurence_angles": cooccurrence_angle or None,
        "difference_variance_interpretation": (
            DifferenceVarianceInterpretation.XMinusYVariance
            if difference_variance == "x-minus-y"
            else DifferenceVarianceInterpretation.ProbabilityXMinusYVariance
        ),
    }
    try:
        if input_path.suffix.lower() == ".csv":
            if mask is not None:
                raise click.UsageError("batch input uses the Mask column; omit --mask")
            write_batch(input_path, outputfile, dimensions, label, extendstats, options)
        else:
            if mask is None:
                raise click.UsageError("--mask is required for image input")
            write_single(
                outputfile, extract(input_path, mask, dimensions, label, extendstats, options)
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    run()
