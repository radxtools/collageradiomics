#!usr/bin/env python
# -*- coding: utf-8 -*-

import click
import os
import sys
import SimpleITK as sitk
import csv
from scipy import stats
import numpy as np

import collageradiomics

@click.command()
@click.option('-i', '--input', required=True, help='Path to an input image from which features will be extracted.')
@click.option('-m', '--mask', required=True, help='Path to a mask that will be considered as binary. The highest pixel value will be considered as information and all other values will be considered outside the mask')
@click.option('-o', '--outputfile', required=True, help='Path to the output CSV file.')
@click.option('-v', '--verbose', default=True, help='Provides additional debug output.')
@click.option('-d', '--dimensions', help='Optional number of dimensions upon which to run collage. Supported values are 2 and 3. If left out, we will default to the dimensionality of the image itself, which may not reflect expected behavior if the image has an alpha channel.', type=click.IntRange(2, 3, clamp=True))
@click.option('-s', '--svdradius', default=5, help='SVD radius is used for the dominant angle calculation pixel radius. DEFAULTS to 5 and is suggested to remain at the default.')
@click.option('-h', '--haralickwindow', default=-1, help='Number of pixels around each pixel used to calculate the haralick texture. DEFAULTS to svdradius * 2 - 1.')
@click.option('-b', '--binsize', default=64, help='Number of bins to use while calculating the grey level cooccurence matrix. DEFAULTS to 64.')
def run(input, mask, outputfile, verbose, dimensions, svdradius, haralickwindow, binsize):
  """CoLlAGe captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood."""
  
  image = sitk.ReadImage(input)
  mask = sitk.ReadImage(mask)

  image_array = sitk.GetArrayFromImage(image)
  mask_array = sitk.GetArrayFromImage(mask)

  # Remove any extra array dimensions if the user explicitly asks for 2D.
  if dimensions == 2:
    image_array = image_array[:,:,0]
    mask_array  = mask_array [:,:,0]

  collage = collageradiomics.Collage(
    image_array, 
    mask_array, 
    svd_radius=svdradius, 
    verbose_logging=verbose,
    num_unique_angles=binsize)

  collage.execute()

  # Create a csv file at the passed in output file location.
  with open(outputfile, 'w', newline='') as csv_output_file:
    writer = csv.writer(csv_output_file)

    # Write the columns.
    writer.writerow(['FeatureName', 'Value'])
    for feature in collageradiomics.HaralickFeature:
      feature_output = collage.get_single_feature_output(feature)
      if image_array.ndim == 2:
        feature_output = feature_output[~np.isnan(feature_output)]

        # NumPy supports median natively, we'll use that.
        median = np.nanmedian(feature_output, axis=None)

        # Use SciPy for kurtosis, variance, and skewness.
        feature_stats = stats.describe(feature_output, axis=None)

        # Write CSV row for current feature.
        _write_csv_stats_row(writer, feature, median, feature_stats.skewness, feature_stats.kurtosis, feature_stats.variance)
      else:
        # Extract phi and theta angles.
        feature_output_theta = feature_output[:,:,:,0]
        feature_output_phi = feature_output[:,:,:,1]

        # Remove NaN for stat calculations.
        feature_output_theta = feature_output_theta[~np.isnan(feature_output_theta)]
        feature_output_phi = feature_output_phi[~np.isnan(feature_output_phi)]

        # NumPy supports median natively, we'll use that.
        median_theta = np.nanmedian(feature_output_theta, axis=None)
        median_phi = np.nanmedian(feature_output_phi, axis=None)

        # Use SciPy for kurtosis, variance, and skewness.
        feature_stats_theta = stats.describe(feature_output_theta.flatten(), axis=None)
        feature_stats_phi = stats.describe(feature_output_phi.flatten(), axis=None)

        # Write CSV rows for each angle.
        _write_csv_stats_row(writer, feature, median_theta, feature_stats_theta.skewness, feature_stats_theta.kurtosis, feature_stats_theta.variance, 'Theta')
        _write_csv_stats_row(writer, feature, median_phi, feature_stats_phi.skewness, feature_stats_phi.kurtosis, feature_stats_phi.variance, 'Phi')

def _write_csv_stats_row(writer, feature, median, skewness, kurtosis, variance, suffix=''):
  writer.writerow([f'Collage{feature.name}Median{suffix}', f'{median:.10f}'])
  writer.writerow([f'Collage{feature.name}Skewness{suffix}', f'{skewness:.10f}'])
  writer.writerow([f'Collage{feature.name}Kurtosis{suffix}', f'{kurtosis:.10f}'])
  writer.writerow([f'Collage{feature.name}Variance{suffix}', f'{variance:.10f}'])

if __name__ == '__main__':
  run()
