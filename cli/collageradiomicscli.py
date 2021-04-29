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
@click.option('-i', '--input', required=True, help='Path to an input image from which features will be extracted or Path to an input csv (header: ID, Image, Mask) from which features will be extracted for several images and masks.')
@click.option('-m', '--mask', help='Path to a mask that will be considered as binary. The highest pixel value will be considered as information and all other values will be considered outside the mask - Required if input is an image')
@click.option('-o', '--outputfile', required=True, help='Path to the output CSV file.')
@click.option('-v', '--verbose', default=True, help='Provides additional debug output.')
@click.option('-d', '--dimensions', help='Optional number of dimensions upon which to run collage. Supported values are 2 and 3. If left out, we will default to the dimensionality of the image itself, which may not reflect expected behavior if the image has an alpha channel.', type=click.IntRange(2, 3, clamp=True))
@click.option('-s', '--svdradius', default=5, help='SVD radius is used for the dominant angle calculation pixel radius. DEFAULTS to 5 and is suggested to remain at the default.')
@click.option('-h', '--haralickwindow', default=-1, help='Number of pixels around each pixel used to calculate the haralick texture. DEFAULTS to svdradius * 2 - 1.')
@click.option('-b', '--binsize', default=64, help='Number of bins to use while calculating the grey level cooccurence matrix. DEFAULTS to 64.')
@click.option('-l', '--label', default=-1, help='Some masks may have multiple labels. Select the label for which to calculate collage radiomics features. Use -1 if you want all labels in the mask as a value of True. DEFAULTS to -1.')
@click.option('-e', '--extendstats', is_flag=True, help='Provides additional statistical metrics (mean and IQR) in the output.')

def run(input, mask, outputfile, verbose, dimensions, svdradius, haralickwindow, binsize,label, extendstats):
  """CoLlAGe captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood."""
  
  if input.endswith('.csv'):
      header = ['ID', 'Image', 'Mask', 'svdradius', 'haralickwindow', 'binsize', 'label']
      features_list = []
      list_failed_cases = [['ID', 'Image', 'Mask', 'Error']]
      if dimensions == 2:
        suffix = ''
        for feature in collageradiomics.HaralickFeature:
          if extendstats:
            features_list.extend(['Collage'+feature.name+'Median'+suffix, 'Collage'+feature.name+'IQR'+suffix, 'Collage'+feature.name+'Skewness'+suffix, 'Collage'+feature.name+'Kurtosis'+suffix, 'Collage'+feature.name+'Mean'+suffix, 'Collage'+feature.name+'Variance'+suffix])
          else:
            features_list.extend(['Collage'+feature.name+'Median'+suffix,  'Collage'+feature.name+'Skewness'+suffix, 'Collage'+feature.name+'Kurtosis'+suffix, 'Collage'+feature.name+'Variance'+suffix])
        header.append(features_list)
        output_list = [header]
      else:
        for suffix in ['Theta', 'Phi']:
          for feature in collageradiomics.HaralickFeature:
            if extendstats:
              features_list.extend(['Collage'+feature.name+'Median'+suffix, 'Collage'+feature.name+'IQR'+suffix, 'Collage'+feature.name+'Skewness'+suffix, 'Collage'+feature.name+'Kurtosis'+suffix, 'Collage'+feature.name+'Mean'+suffix, 'Collage'+feature.name+'Variance'+suffix])
            else:
              features_list.extend(['Collage'+feature.name+'Median'+suffix, 'Collage'+feature.name+'Skewness'+suffix, 'Collage'+feature.name+'Kurtosis'+suffix, 'Collage'+feature.name+'Variance'+suffix])
        header.extend(features_list)
        output_list = [header]
      
      with open(input, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
          output_case = []
          try:
            case_id = row['ID']
            image_filepath = row['Image']
            mask_filepath = row['Mask']
            image = sitk.ReadImage(image_filepath)
            mask = sitk.ReadImage(mask_filepath)
            
            output_case.extend([case_id, image_filepath, mask_filepath, svdradius, haralickwindow, binsize, label])
            
            # Check if user wants to select single label from the mask
            if label != -1:
              mask = sitk.BinaryThreshold(mask, lowerThreshold = label, upperThreshold = label, insideValue = 1, outsideValue = 0)
            
            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)
            
            # Collage is expecting array with x,y,z but sitk.GetArrayFromImage as z,y,x, so x show be swapped by z
            if dimensions != 2:
              image_array = np.swapaxes(image_array,0,2)
              mask_array = np.swapaxes(mask_array,0,2)
            
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
            
            for feature in collageradiomics.HaralickFeature:
              feature_output = collage.get_single_feature_output(feature)
              if image_array.ndim == 2:
                feature_output = feature_output[~np.isnan(feature_output)]

                # NumPy supports median natively, we'll use that.
                median = np.nanmedian(feature_output, axis=None)

                # Use SciPy for kurtosis, variance, and skewness.
                feature_stats = stats.describe(feature_output, axis=None)
                
                if extendstats:
                  mean = feature_stats.mean #np.nanmean(feature_output, axis=None)
                  iqr = stats.iqr(feature_output)
                  
                  output_case.extend([median, iqr, feature_stats.skewness, feature_stats.kurtosis, feature_stats.mean, feature_stats.variance])
                else:
                  output_case.extend([median, feature_stats.skewness, feature_stats.kurtosis, feature_stats.variance])
                
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
                
                if extendstats:
                  mean_theta = feature_stats_theta.mean
                  mean_phi = feature_stats_phi.mean
                  iqr_theta = stats.iqr(feature_output_theta)
                  iqr_phi = stats.iqr(feature_output_phi)
                  
                  output_case.extend([median_theta, iqr_theta, feature_stats_theta.skewness, feature_stats_theta.kurtosis, feature_stats_theta.mean, feature_stats_theta.variance, median_phi, iqr_phi, feature_stats_phi.skewness, feature_stats_phi.kurtosis, feature_stats_phi.mean, feature_stats_phi.variance])
                else:
                  output_case.extend([median_theta, feature_stats_theta.skewness, feature_stats_theta.kurtosis, feature_stats_theta.variance, median_phi, feature_stats_phi.skewness, feature_stats_phi.kurtosis, feature_stats_phi.variance])
            output_list.append(output_case)
          except RuntimeError as err:
            list_failed_cases.append([case_id, image_filepath, mask_filepath, err])
          except ValueError as err:
            list_failed_cases.append([case_id, image_filepath, mask_filepath, err])
      
      # Create collage radiomic features output csv file
      with open(outputfile, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(output_list)
      
      # Create errors output csv file
      with open(os.path.join(os.path.dirname(outputfile), 'errors_' + os.path.basename(outputfile)), 'w') as file:
        writer = csv.writer(file)
        writer.writerows(list_failed_cases)
  else:
      image = sitk.ReadImage(input)
      mask = sitk.ReadImage(mask)
            
      # Check if user wants to select single label from the mask
      if label != -1:
        mask = sitk.BinaryThreshold(mask, lowerThreshold = label, upperThreshold = label, insideValue = 1, outsideValue = 0)

      image_array = sitk.GetArrayFromImage(image)
      mask_array = sitk.GetArrayFromImage(mask)
            
      # Collage is expecting array with x,y,z but sitk.GetArrayFromImage as z,y,x, so x show be swapped by z
      if dimensions != 2:
        image_array = np.swapaxes(image_array,0,2)
        mask_array = np.swapaxes(mask_array,0,2)

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
            
            if extendstats:
                mean_phi = feature_stats_phi.mean
                iqr_phi = stats.iqr(feature_output_phi.flatten())
                
                mean_theta = feature_stats_theta.mean
                iqr_theta = stats.iqr(feature_output_theta.flatten())
                
                _write_csv_extented_stats_row(writer, feature, median_theta, iqr_theta, feature_stats_theta.skewness, feature_stats_theta.kurtosis, mean_theta, feature_stats_theta.variance, 'Theta')
                _write_csv_extented_stats_row(writer, feature, median_phi, iqr_phi, feature_stats_phi.skewness, feature_stats_phi.kurtosis, mean_phi, feature_stats_phi.variance, 'Phi')
            else:
                # Write CSV rows for each angle.
                _write_csv_stats_row(writer, feature, median_theta, feature_stats_theta.skewness, feature_stats_theta.kurtosis, feature_stats_theta.variance, 'Theta')
                _write_csv_stats_row(writer, feature, median_phi, feature_stats_phi.skewness, feature_stats_phi.kurtosis, feature_stats_phi.variance, 'Phi')

def _write_csv_stats_row(writer, feature, median, skewness, kurtosis, variance, suffix=''):
  writer.writerow([f'Collage{feature.name}Median{suffix}', f'{median:.10f}'])
  writer.writerow([f'Collage{feature.name}Skewness{suffix}', f'{skewness:.10f}'])
  writer.writerow([f'Collage{feature.name}Kurtosis{suffix}', f'{kurtosis:.10f}'])
  writer.writerow([f'Collage{feature.name}Variance{suffix}', f'{variance:.10f}'])
  
def _write_csv_extented_stats_row(writer, feature, median, iqr, skewness, kurtosis, mean, variance, suffix=''):
  writer.writerow([f'Collage{feature.name}Median{suffix}', f'{median:.10f}'])
  writer.writerow([f'Collage{feature.name}IQR{suffix}', f'{iqr:.10f}'])
  writer.writerow([f'Collage{feature.name}Skewness{suffix}', f'{skewness:.10f}'])
  writer.writerow([f'Collage{feature.name}Kurtosis{suffix}', f'{kurtosis:.10f}'])
  writer.writerow([f'Collage{feature.name}Mean{suffix}', f'{mean:.10f}'])
  writer.writerow([f'Collage{feature.name}Variance{suffix}', f'{variance:.10f}'])

if __name__ == '__main__':
  run()
