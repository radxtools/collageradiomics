#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import SimpleITK as sitk

# import collagaradiomics


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='Collageradiomics_CLI', formatter_class=argparse.RawTextHelpFormatter,
                                    description='\nThis code is used to get the Collage features. ')

  parser.add_argument('-i', dest='inputImage', type=str,
                      help='The single input image to extract features from.\n',
                      required=True)
  
  parser.add_argument('-m', dest='binaryMask', type=str,
                      help='The binary mask containing annotation.\n',
                      required=True)

  parser.add_argument('-o', dest='outputCSV', type=str,
                      help='The output CSV file.\n',
                      required=True)

  # parser.add_argument('-v', '--version', action='version',
  #                     version=pkg_resources.require("Hausdorff95")[0].version, help="Show program's version number and exit.") # disabled because pyinstaller doesn't import it properly
                          
  args = parser.parse_args()

  inputImage = os.path.abspath(args.inputImage)
  maskImage = os.path.abspath(args.maskImage)
  outputCSV = os.path.abspath(args.outputCSV)

  image = sitk.GetArrayFromImage(sitk.ReadImage(inputImage))
  mask = sitk.GetArrayFromImage(sitk.ReadImage(maskImage))

  dim = image.GetDimension()
  # sanity check
  if image.GetDimension() != mask.GetDimension():
    sys.exit('The Dimension of input image and mask need to be the same')

  ## todo: extract collage features here
  if dim == 2:
    test = 2 # extract 2D collage features 
  elif dim == 3:
    test = 3 # extract 3D collage features 

  ## todo: write CSV file here
