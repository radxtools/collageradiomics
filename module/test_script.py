import collageradiomics
import pydicom
import logging
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from skimage.exposure import equalize_hist
import numpy as np
from sklearn.preprocessing import minmax_scale
from random import randint

level = logging.INFO
logging.basicConfig(level=level)
logger = logging.getLogger()
logger.setLevel(level)
logger.info('Hello, world.')

local_dcm_file = 'test.dcm'
instance = pydicom.dcmread(local_dcm_file)
slice_instance_uid = instance.SOPInstanceUID
logger.debug(f'slice_instance_uid  = {slice_instance_uid}')

logger.info('Correcting image...')
np_array = instance.pixel_array
corrected = apply_modality_lut(np_array, instance)
corrected = apply_voi_lut(corrected, instance)
logger.debug(f'np.histogram(scaled_array) = {np.histogram(corrected)}')
scaled_array = equalize_hist(corrected)
logger.debug(f'np.histogram(scaled_array) = {np.histogram(scaled_array)}')
logger.info('done.')

width = 50
height = 50
min_row = randint(30,300)
max_row = min_row + height
min_col = randint(30,300)
max_col = min_col + width

original_shape = np_array.shape
logger.debug(f'original_shape = {original_shape}')

logger.info('Calculating collage features...')
mask_array = np.zeros(original_shape, dtype='int')
mask_array[min_row:max_row, min_col:max_col] = 1
textures = collageradiomics.Collage(scaled_array, mask_array).execute()
logger.info('Collage features calculated.')

logger.debug(f'textures.shape = {textures.shape}')
logger.debug(f'textures.dtype = {textures.dtype}')
logger.debug(f'np.histogram(textures) = {np.histogram(textures, range=(np.nanmin(textures), np.nanmax(textures)))}')

logger.info('Defining DICOM bit data to store unsigned greyscale texture output:')
# http://dicomiseasy.blogspot.com/2012/08/chapter-12-pixel-data.html
texture_bit_depth = 16
texture_dtype = np.uint16

instance.PhotometricInterpretation = 'MONOCHROME2'
instance.SamplesPerPixel = 1
instance.BitsAllocated = texture_bit_depth
instance.BitsStored = texture_bit_depth
instance.HighBit = texture_bit_depth - 1
instance.PixelRepresentation = 0 # unsigned
logger.info('DICOM bit data defined.')

min_output_value = 0
max_output_value = 2**texture_bit_depth-1

for texture_index in range(textures.shape[2]):
    # https://stackoverflow.com/a/65964648
    texture_slice = textures[:,:,texture_index]
    #logger.debug(np.histogram(texture_slice, range=(np.nanmin(texture_slice), np.nanmax(texture_slice))))
    
    logger.info('Rescaling texture to full range of DICOM bit depth:')
    flattened = texture_slice.flatten()
    scaled = minmax_scale(flattened, (min_output_value, max_output_value))
    texture_slice = scaled.reshape(texture_slice.shape)
    #logger.debug(np.histogram(texture_slice, range=(np.nanmin(texture_slice), np.nanmax(texture_slice))))
    logger.info('Rescaling texture to full range of DICOM bit depth done.')
    
    logger.info('Casting to {texture_dtype} and storing in instance...')
    output_pixel_data = texture_slice.astype(texture_dtype)
    #logger.debug(np.histogram(output_pixel_data, range=(np.nanmin(output_pixel_data), np.nanmax(output_pixel_data))))
    #logger.debug(f'output_pixel_data.dtype = {output_pixel_data.dtype}')
    instance.PixelData = output_pixel_data
    logger.info('Storing in instance done.')
