import math
import random
import sys, select
from itertools import product

import mahotas as mt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg
from skimage.feature.texture import greycomatrix
from skimage.util.shape import view_as_windows


def svd_dominant_angle(x, y, dx_windows, dy_windows):
    """Calculates the dominate angle at the coordinate within the windows.


        :param x: x value of coordinate
        :type x: int
        :param y: y value of coordinate
        :type y: int
        :param dx_windows: dx windows of x, y shape to run svd upon
        :type dx_windows: numpy.ndarray
        :param dy_windows: dy windows of x, y shape to run svd upon
        :type dy_windows: numpy.ndarray

        :returns: dominant angle at x, y
        :rtype: float 
    """
    dx_patch = dx_windows[y, x]
    dy_patch = dy_windows[y, x]

    window_area = dx_patch.size
    flattened_gradients = np.zeros((window_area, 2))
    flattened_gradients[:, 0] = np.reshape(dx_patch, ((window_area)), order='F')
    flattened_gradients[:, 1] = np.reshape(dy_patch, ((window_area)), order='F')

    _, _, v = linalg.svd(flattened_gradients)
    dominant_angle = math.atan2(v[0, 0], v[1, 0])

    return dominant_angle


def show_colored_image(figure, axis, image_data, colormap=plt.cm.jet):
    """Helper method to show a colored image in matplotlib.


        :param figure: figure upon which to display
        :type figure: matplotlib.figure.Figure
        :param axis: axis upon which to display
        :type axis: matplotlib.axes.Axes
        :param image_data: image to display
        :type image_data: numpy.ndarray
        :param colormap: color map to convert for display. Defaults to plt.cm.jet.
        :type colormap: matplotlib.colors.Colormap, optional
    """
    image = axis.imshow(image_data, cmap=colormap)
    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(image, cax=colorbar_axis)


def create_highlighted_rectangle(x, y, w, h):
    """Creates a matplotlib Rectangle object for a highlight effect


        :param x: x location to start rectangle
        :type x: int
        :param y: y location to start rectangle
        :type y: int
        :param w: width of rectangle
        :type w: int
        :param h: height of rectangle
        :type h: int

        :returns: Rectangle used to highlight within a plot
        :rtype: matplotlib.patches.Rectangle
    """
    return Rectangle((x, y), w, h, linewidth=3, edgecolor='cyan', facecolor='none')


def highlight_rectangle_on_image(image_data, min_x, min_y, w, h, colormap=plt.cm.gray):
    """Highlights a rectangle on an image at the passed in coordinate.


        :param image_data: image to highlight
        :type image_data: numpy.ndarray
        :param min_x: x location to start highlight
        :type min_x: int
        :param min_y: y location to start highlight
        :type min_y: int
        :param w: width of highlight rectangle
        :type w: int
        :param h: height of highlight rectangle
        :type h: int
        :param colormap: color map to convert for display. Defaults to plt.cm.jet.
        :type colormap: matplotlib.colors.Colormap, optional

        :returns: image array with highlighted rectangle
        :rtype: numpy.ndarray
    """
    figure, axes = plt.subplots(1, 2, figsize=(15, 15))

    # Highlight window within image.
    show_colored_image(figure, axes[0], image_data, colormap)
    axes[0].add_patch(create_highlighted_rectangle(min_x, min_y, w, h))

    # Crop window.
    cropped_array = image_data[min_y:min_y + h, min_x:min_x + w]
    axes[1].set_title(f'Cropped Region ({w}x{h})')
    show_colored_image(figure, axes[1], cropped_array, colormap)

    plt.show()

    return cropped_array


def scale_array_for_image(array_to_scale):
    """Scales an array from 0-255 integer values


        :param array_to_scale: array to scale
        :type array_to_scale: numpy.ndarray

        :returns: array scaled from 0-255
        :rtype: numpy.ndarray
    """
    flat_array = array_to_scale.flatten()
    minimum = float(min(flat_array))
    maximum = float(max(flat_array))
    array_range = maximum - minimum
    array_to_scale = array_to_scale - minimum
    array_to_scale /= array_range
    array_to_scale *= 255
    return array_to_scale


from enum import Enum


class HaralickFeature(Enum):
    """Enumeration Helper For Haralick Features


        :param Enum: Enumeration Helper For Haralick Features
        :type Enum: HaralickFeature
    """
    AngularSecondMoment = 0
    Contrast = 1
    Correlation = 2
    SumOfSquareVariance = 3
    SumAverage = 4
    SumVariance = 5
    SumEntropy = 6
    Entropy = 7
    DifferenceVariance = 8
    DifferenceEntropy = 9
    InformationMeasureOfCorrelation1 = 10
    InformationMeasureOfCorrelation2 = 11
    MaximalCorrelationCoefficient = 12
    All = 13


class DifferenceVarianceInterpretation(Enum):
    """ Feature 10 has two interpretations, as the variance of |x-y|
        or as the variance of P(|x-y|).
        See: https://ieeexplore.ieee.org/document/4309314

        :param Enum: Enumeration Helper For Haralick Features
        :type Enum: DifferenceVarianceInterpretation
    """
    XMinusYVariance = 0
    ProbabilityXMinusYVariance = 1


class CollageCollection:
    """ Enables computation on multiple collage images at the same time.
        See :py:meth:`from_multiple_images` function for example usage.
    """

    def __init__(self,
                 images_array,
                 masks_array,
                 svd_radius=5,
                 verbose_logging=False,
                 haralick_feature_list=[HaralickFeature.All],
                 log_sample_rate=500,
                 cooccurence_angles=[0, 1 * np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4, 4 * np.pi / 4, 5 * np.pi / 4,
                                     6 * np.pi / 4, 7 * np.pi / 4],
                 difference_variance_interpretation=DifferenceVarianceInterpretation.XMinusYVariance,
                 haralick_window_size=-1,
                 greylevels=64,
                 ):
        """Designated initializer for CollageCollection

            :param images_array: list of images to run collage upon
            :type images_array: [numpy.ndarray]
            :param mask_array: list of masks that correlates with the images
            :type mask_array: [numpy.ndarray]
            :param svd_radius: radius of svd. Defaults to 5.
            :type svd_radius: int, optional
            :param verbose_logging: turning this on will log intermediate results]. Defaults to False.
            :type verbose_logging: bool, optional
            :param haralick_feature_list: array of features to calculate. Defaults to [HaralickFeature.All].
            :type haralick_feature_list: [HaralickFeature], optional
            :param log_sample_rate: higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
            :type log_sample_rate: int, optional
            :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
            :type cooccurence_angles: list, optional
            :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
            :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
            :type haralick_window_size: int, optional
            :param greylevels: number of bins to use for the texture calculation. Defaults to 64.
            :type greylevels: int, optional
        """
        self._images_array = images_array
        self._masks_array = masks_array
        self._svd_radius = svd_radius
        self._verbose_logging = verbose_logging
        self._haralick_feature_list = haralick_feature_list
        self._log_sample_rate = log_sample_rate
        self._cooccurence_angles = cooccurence_angles
        self._difference_variance_interpretation = difference_variance_interpretation
        self._haralick_window_size = haralick_window_size
        self._greylevels = greylevels
        collages = []
        for x in range(len(images_array)):
            mask_index = 0
            if len(masks_array) >= x - 1:
                mask_index = x
            collage = Collage(
                images_array[x],
                masks_array[mask_index],
                svd_radius,
                verbose_logging,
                haralick_feature_list,
                log_sample_rate,
                cooccurence_angles,
                difference_variance_interpretation,
                haralick_window_size,
                greylevels
            )
            collages.append(collage)
        self.collages = collages

    def execute(self):
        for collage in self.collages:
            collage.execute()


class Collage:
    """This is the main object in the Collage calculation system. Usage: create a Collage object and then call the :py:meth:`execute` function.

        :param image_array: image to run collage upon
        :type image_array: numpy.ndarray
        :param mask_array: mask that correlates with the image
        :type mask_array: numpy.ndarray
        :param svd_radius: radius of svd. Defaults to 5.
        :type svd_radius: int, optional
        :param verbose_logging: turning this on will log intermediate results]. Defaults to False.
        :type verbose_logging: bool, optional
        :param haralick_feature_list: array of features to calculate. Defaults to [HaralickFeature.All].
        :type haralick_feature_list: [HaralickFeature], optional
        :param log_sample_rate: higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
        :type log_sample_rate: int, optional
        :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
        :type cooccurence_angles: list, optional
        :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
        :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
        :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
        :type haralick_window_size: int, optional
        :param greylevels: number of bins to use for the texture calculation. Defaults to 64.
        :type greylevels: int, optional
    """


    @property
    def img_array(self):
        """
        The original image.

        :getter: Returns the original image array.
        :setter: Sets the original image array.
        :type: np.ndarray
        """
        return self._img_array

    @property
    def mask_array(self):
        """
        Array passed into Collage.

        :getter: Returns the original mask array.
        :setter: Sets the original mask array.
        :type: np.ndarray
        """        
        return self._mask_array

    @property
    def is_3D(self):
        """
        Whether we are using 3D collage calculations (True) or 2D (False)
        """
        return self._is_3D
        
    @property
    def svd_radius(self):
        """
        SVD radius is used to calculate the pixel radius
        for the dominant angle calculation.

        :getter: Returns the SVD radius.
        :setter: Sets the SVD radius.
        :type: int
        """   
        return self._svd_radius

    @property
    def verbose_logging(self):
        """
        Turning this on will result in more
        detailed logging.

        :getter: Returns True if on.
        :setter: Turns verbose logging off or on.
        :type: bool
        """
        return self._verbose_logging

    @property
    def haralick_feature_list(self):
        """
        Iterable of which haralick features to calculate.

        :getter: Returns True if on.
        :setter: Turns verbose logging off or on.
        :type: [HaralickFeature]
        """
        return self._haralick_feature_list

    @property
    def log_sample_rate(self):
        """
        Sample rate of how many angles we log.

        :getter: Returns the sample rate.
        :setter: A higher value will result in more logging.
        :type: int
        """
        return self._log_sample_rate

    @property
    def cooccurence_angles(self):
        """
        Iterable of angles that will be used in the cooccurence matrix.

        :getter: Returns the Iterable of cooccurence angles.
        :setter: Sets the angles to be used in the cooccurence matrix.
        :type: int
        """
        return self._cooccurence_angles

    @property
    def difference_variance_interpretation(self):
        """
        Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].
        Defaults to DifferenceVarianceInterpretation.XMinusYVariance.

        :getter: Returns requested variance interpretation.
        :setter: Sets requested variance interpretation.
        :type: DifferenceVarianceInterpretation
        """
        return self._difference_variance_interpretation

    @property
    def haralick_window_size(self):
        """
        Number of pixels around each pixel to calculate a haralick texture.

        :getter: Returns requested number of pixels.
        :setter: Sets requested number of pixels.
        :type: int
        """
        return self._haralick_window_size

    @property
    def greylevels(self):
        """
        Number of bins to use for texture calculations. Defaults to 64.

        :getter: Returns requested number of bins.
        :setter: Sets requested number of bins.
        :type: int
        """
        return self._greylevels

    @property
    def full_images(self):
        """
        numpy.ndarray image representing collage upon the mask within the full images.

        :getter: Returns original image with collage upon the mask region.
        :setter: Sets original image with collage upon the mask.
        :type: numpy.ndarray
        """
        return self._full_images

    @full_images.setter
    def full_images(self, value):
        self._full_images = value

    @property
    def full_masked_images(self):
        """
        numpy.ndarray image representing collage upon the mask within the full images. 
        Does not include values outside the mask.

        :getter: Returns images with empty values outside of mask and with collage upon the mask region.
        :setter: Sets images with empty values outside of mask and with collage upon the mask region.
        :type: numpy.ndarray
        """
        return self._full_masked_images

    @full_masked_images.setter
    def full_masked_images(self, value):
        self._full_masked_images = value

    @property
    def haralick_feature_list(self):
        """
        Iterable representing the list of requested features.

        :getter: Returns the list of requested features.
        :setter: Sets the list of requested features.
        :type: Iterable
        """
        return self._haralick_feature_list

    @haralick_feature_list.setter
    def haralick_feature_list(self, value):
        self._haralick_feature_list = value

    def __init__(self,
                 img_array,
                 mask_array,
                 svd_radius=5,
                 verbose_logging=False,
                 haralick_feature_list=[HaralickFeature.All],
                 log_sample_rate=500,
                 cooccurence_angles=[0, 1 * np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4, 4 * np.pi / 4, 5 * np.pi / 4,
                                     6 * np.pi / 4, 7 * np.pi / 4],
                 difference_variance_interpretation=DifferenceVarianceInterpretation.XMinusYVariance,
                 haralick_window_size=-1,
                 greylevels=64,
                 ):
        """Designated initializer for Collage

            :param image_array: image to run collage upon
            :type image_array: numpy.ndarray
            :param mask_array: mask that correlates with the image
            :type mask_array: numpy.ndarray
            :param svd_radius: radius of svd. Defaults to 5.
            :type svd_radius: int, optional
            :param verbose_logging: turning this on will log intermediate results. Defaults to False.
            :type verbose_logging: bool, optional
            :param haralick_feature_list: array of features to calculate. Defaults to [HaralickFeature.All].
            :type haralick_feature_list: [HaralickFeature], optional
            :param log_sample_rate: higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
            :type log_sample_rate: int, optional
            :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
            :type cooccurence_angles: list, optional
            :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
            :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
            :type haralick_window_size: int, optional
            :param greylevels: number of bins to use for the texture calculation. Defaults to 64.
            :type greylevels: int, optional
        """
        
        if verbose_logging:
            print('Collage Module Reloaded')
        
        if haralick_window_size == -1:
            self._haralick_window_size = svd_radius * 2 + 1
        else:
            self._haralick_window_size = haralick_window_size

        if self._haralick_window_size < 1:
            raise Exception('Haralick windows size must be at least 1 pixel.')

        if svd_radius < 1:
            raise Exception('SVD radius must be at least 1 pixel')

        if greylevels < 1:
            raise Exception('greylevels must contain at least 1 bin')

        if img_array.ndim < 2 or img_array.ndim > 3:
            raise Exception('Expected a 2D or 3D image.')

        if mask_array.shape != img_array.shape:
            raise Exception('Mask must be the same shape as image.')
        
        self._img_array = img_array
        
        self._is_3D = img_array.ndim == 3
        if verbose_logging:
            print(f'Running 3D Collage = {self.is_3D}')

        min_3D_slices = 3;
        if self._img_array.shape[0] <  self._haralick_window_size or self._img_array.shape[1] < self._haralick_window_size or (self._is_3D and self._img_array.shape[2] < min_3D_slices):
            raise Exception(
                f'Image is too small for a window size of {self._haralick_window_size} pixels.')
        
        uniqueValues = np.unique(mask_array)
        numberOfValues = len(uniqueValues)
        if numberOfValues > 2:
            if verbose_logging:
                for color in uniqueValues:
                    print(f'Found color value of {color}.')
            print(f'Mask is not binary, there are {numberOfValues} unique colors in the image.')
            print(f'Continuing with mask of {uniqueValues.max()}')
        trimmed_mask_array = (mask_array == uniqueValues.max()).astype('float64')
        non_zero_indices = np.argwhere(trimmed_mask_array)
        try:
            (min_y, min_x), (max_y, max_x) = non_zero_indices.min(0), non_zero_indices.max(0) + 1
        except:
            raise Exception('Non-contiguous masks are not supported.')
        self.mask_min_x = min_x
        self.mask_min_y = min_y
        self.mask_max_x = max_x
        self.mask_max_y = max_y

        scaled_mask_array = mask_array[self.mask_min_y:self.mask_max_y, self.mask_min_x:self.mask_max_x]
        self.mask_width = self.mask_max_x - self.mask_min_x
        self.mask_height = self.mask_max_y - self.mask_min_y
        self._svd_radius = svd_radius
        self._verbose_logging = verbose_logging
        self._mask_array = scaled_mask_array

        self._haralick_feature_list = haralick_feature_list
        self._feature_count = len(haralick_feature_list)
        self._log_sample_rate = log_sample_rate
        self._cooccurence_angles = cooccurence_angles
        self._difference_variance_interpretation = difference_variance_interpretation

        self._greylevels = greylevels

    @classmethod
    def from_rectangle(cls,
                       img_array,
                       mask_min_x,
                       mask_min_y,
                       mask_width,
                       mask_height,
                       svd_radius=5,
                       verbose_logging=False,
                       haralick_feature_list=[HaralickFeature.All],
                       log_sample_rate=500,
                       cooccurence_angles=[0, 1 * np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4, 4 * np.pi / 4, 5 * np.pi / 4,
                                           6 * np.pi / 4, 7 * np.pi / 4],
                       difference_variance_interpretation=DifferenceVarianceInterpretation.XMinusYVariance,
                       haralick_window_size=-1,
                       greylevels=64,
                       ):
        """Creates a rectangle at the x,y,w,h and only calculates collage inside that mask.

            :param img_array: image to run collage upon
            :type img_array: numpy.ndarray
            :param mask_min_x: x location of window
            :type mask_min_x: int
            :param mask_min_y: y location of window
            :type mask_min_y: int
            :param mask_width: window width
            :type mask_width: int
            :param mask_height: window height
            :type mask_height: int
            :param svd_radius: radius of svd. Defaults to 5.
            :type svd_radius: int, optional
            :param verbose_logging: turning this on will log intermediate results. Defaults to False.
            :type verbose_logging: bool, optional
            :param haralick_feature_list: array of features to calculate. Defaults to [HaralickFeature.All].
            :type haralick_feature_list: [HaralickFeature], optional
            :param log_sample_rate: higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
            :type log_sample_rate: int, optional
            :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
            :type cooccurence_angles: list, optional
            :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
            :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
            :type haralick_window_size: int, optional
            :param greylevels: number of bins to use for the texture calculation. Defaults to 64.
            :type greylevels: int, optional
            :returns: Collage object to run collage on a rectangular section of the image.
            :rtype: Collage
        """
        mask_array = np.zeros((img_array.shape[0], img_array.shape[1]))
        mask_array[mask_min_y:mask_min_y + mask_height, mask_min_x:mask_min_x + mask_width] = 255
        return cls(
            img_array,
            mask_array,
            svd_radius,
            verbose_logging,
            haralick_feature_list,
            log_sample_rate,
            cooccurence_angles,
            difference_variance_interpretation,
            haralick_window_size,
            greylevels
        )

    @classmethod
    def from_multiple_images(cls,
                             images_array,
                             masks_array,
                             svd_radius=5,
                             verbose_logging=False,
                             haralick_feature_list=[HaralickFeature.All],
                             log_sample_rate=500,
                             cooccurence_angles=[0, 1 * np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4, 4 * np.pi / 4,
                                                 5 * np.pi / 4, 6 * np.pi / 4, 7 * np.pi / 4],
                             difference_variance_interpretation=DifferenceVarianceInterpretation.XMinusYVariance,
                             haralick_window_size=-1,
                             greylevels=64,
                             ):
        """Helper method to run collage on multiple images.

            :param images_array: list of images to run collage upon
            :type images_array: [numpy.ndarray]
            :param mask_array: list of masks that correlates with the images
            :type mask_array: [numpy.ndarray]
            :param svd_radius: radius of svd. Defaults to 5.
            :type svd_radius: int, optional
            :param verbose_logging: turning this on will log intermediate results]. Defaults to False.
            :type verbose_logging: bool, optional
            :param haralick_feature_list: array of features to calculate. Defaults to [HaralickFeature.All].
            :type haralick_feature_list: [HaralickFeature], optional
            :param log_sample_rate: higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
            :type log_sample_rate: int, optional
            :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
            :type cooccurence_angles: list, optional
            :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
            :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
            :type haralick_window_size: int, optional
            :param greylevels: number of bins to use for the texture calculation. Defaults to 64.
            :type greylevels: int, optional

            :returns: CollageCollection object that works similarly to Collage. Call execute() to run on all images.
            :rtype: CollageCollection
        """
        return CollageCollection(
            images_array,
            masks_array,
            svd_radius,
            verbose_logging,
            haralick_feature_list,
            log_sample_rate,
            cooccurence_angles,
            difference_variance_interpretation,
            haralick_window_size,
            greylevels
        )

    def get_haralick_value(self, img_array, center_x, center_y, window_size, greylevels, haralick_feature, symmetric,
                              mean):
                              
        """Gets the haralick texture value at the center of an x, y coordinate.

            :param image_array: image to calculate texture
            :type image_array: numpy.ndarray
            :param center_x: x center of coordinate
            :type center_x: int
            :param center_y: y center of coordinate
            :type center_y: int
            :param window_size: size of window to pull for calculation
            :type window_size: int
            :param greylevels: number of bins
            :type greylevels: int
            :param haralick_feature: desired haralick feature
            :type haralick_feature: HaralickFeature
            :param symmetric: whether or not we should use the symmetrical cooccurence matrix
            :type symmetric: bool
            :param mean: whether we return the mean of the feature or not
            :type mean: bool

            :returns: number representing value of haralick texture at coordinate.
            :rtype: float
        """
        # extract subpart of image (todo: pass in result from view_as_windows)
        min_x = int(max(0, center_x - window_size / 2 - 1))
        min_y = int(max(0, center_y - window_size / 2 - 1))
        max_x = int(min(img_array.shape[1] - 1, center_x + window_size / 2 + 1))
        max_y = int(min(img_array.shape[0] - 1, center_y + window_size / 2 + 1))
        cropped_img_array = img_array[min_y:max_y, min_x:max_x]

        # co-occurence matrix of all 8 directions and sum them
        cooccurence_matrix = greycomatrix(cropped_img_array, [1], self.cooccurence_angles, levels=greylevels)
        cooccurence_matrix = np.sum(cooccurence_matrix, axis=3)
        cooccurence_matrix = cooccurence_matrix[:, :, 0]

        # extract haralick using mahotas library:
        har_feature = mt.features.texture.haralick_features([cooccurence_matrix], return_mean=mean)

        # output:
        if mean:
            return har_feature[haralick_feature]
        return har_feature[0, haralick_feature]

    def get_haralick_feature(self, img, desired_haralick_feature, greylevels, haralick_window_size, symmetric=False,
                                mean=False):
        """Gets haralick image within the mask.

            :param img: image to get feature from
            :type img: numpy.ndarray
            :param desired_haralick_feature: which feature to calculate
            :type desired_haralick_feature: Haralick Feature
            :param greylevels: number of bins
            :type greylevels: int
            :param haralick_window_size: size of window around pixels to calculate haralick value
            :type haralick_window_size: int
            :param symmetric: (bool, optional)
                whether or not we should use the symmetrical cooccurence matrix. 
                Defaults to False.
            mean (bool, optional)
                whether we return the mean of the feature or not. 
                Defaults to False.

            :returns: An image representing haralick texture.
            :rtype: numpy.ndarray
        """
        haralick_image = np.zeros(img.shape)
        h, w = img.shape
        for pos in product(range(w), range(h)):
            if self.mask_array[pos[1]][pos[0]] != 0:
                result = self.get_haralick_value(img, pos[0], pos[1], haralick_window_size, greylevels,
                                                    desired_haralick_feature, symmetric, mean)
                haralick_image[pos[1], pos[0]] = result
        return haralick_image

    def execute(self):
        """Begins haralick calculation.

            :returns: An image at original size that only has the masked section filled in with collage calculations.
            :rtype: numpy.ndarray
        """
        mask_min_x = int(self.mask_min_x)
        mask_min_y = int(self.mask_min_y)
        mask_max_x = int(self.mask_max_x)
        mask_max_y = int(self.mask_max_y)
        mask_width = int(self.mask_width)
        mask_height = int(self.mask_height)
        svd_radius = self.svd_radius
        is_3D = False
        if (self.img_array.ndim == 2):
            img_array = self.img_array
        elif (self.img_array.ndim == 3):
            is_3D = True
            img_array = self.img_array[:, :, 0]
        else:
            raise IndexError('Expected 2D or 3D numpy array.')
        
        if self.verbose_logging:
            print(f'IMAGE:\nwidth={img_array.shape[1]} height={img_array.shape[0]}')

        cropped_array = img_array[mask_min_y:mask_min_y + mask_height,
                        mask_min_x:mask_min_x + mask_width]
        if self.verbose_logging:
            print(f'Cropped Array Shape: {cropped_array.shape}')

        # Extend outwards
        padded_mask_min_x = max(mask_min_x - svd_radius, 0)
        padded_mask_min_y = max(mask_min_y - svd_radius, 0)
        padded_mask_max_x = min(mask_max_x + svd_radius, img_array.shape[1] - 1)
        padded_mask_max_y = min(mask_max_y + svd_radius, img_array.shape[0] - 1)
        if self.verbose_logging:
            print(f'x = {padded_mask_min_x}:{padded_mask_max_x} ({padded_mask_max_x - padded_mask_min_x})')
            print(f'y = {padded_mask_min_y}:{padded_mask_max_y} ({padded_mask_max_y - padded_mask_min_y})')
        padded_cropped_array = img_array[padded_mask_min_y:padded_mask_max_y, padded_mask_min_x:padded_mask_max_x]
        if self.verbose_logging:
            print(f'PaddedCropped Array Shape: {padded_cropped_array.shape}')

        # Calculate gradient
        rescaled_padded_cropped_array = padded_cropped_array / 256
        dx = np.gradient(rescaled_padded_cropped_array, axis=1)
        dy = np.gradient(rescaled_padded_cropped_array, axis=0)
        self.dx = dx
        self.dy = dy

        # loop through all regions and calculate dominant angles

        dominant_angles_array = np.zeros((mask_height, mask_width), np.single)

        if self.verbose_logging:
            print(f'dx shape = {dx.shape}')
            print(f'dominant angles shape = {dominant_angles_array.shape}')

        svd_diameter = svd_radius * 2 + 1
        dx_windows = view_as_windows(dx, (svd_diameter, svd_diameter))
        dy_windows = view_as_windows(dy, (svd_diameter, svd_diameter))

        if self.verbose_logging:
            print(f'svd radius = {svd_radius}')
            print(f'svd diameter = {svd_diameter}')
            print(f'dx windows shape = {dx_windows.shape}')

        center_x_range = range(dx_windows.shape[1])
        center_y_range = range(dx_windows.shape[0])

        if self.verbose_logging:
            print(f'Center x: {center_x_range}, Center y: {center_y_range}')
        for current_svd_center_x in center_x_range:
            for current_svd_center_y in center_y_range:
                current_dominant_angle = svd_dominant_angle(
                    current_svd_center_x, current_svd_center_y,
                    dx_windows, dy_windows)
                dominant_angles_array[current_svd_center_y, current_svd_center_x] = current_dominant_angle
                if random.randint(0, self.log_sample_rate) == 0:
                    if self.verbose_logging:
                        print(f'x={current_svd_center_x}, y={current_svd_center_y}')
                        print(f'angle={current_dominant_angle}')

        if self.verbose_logging:
            print('Done calculating dominant angles.')

        self.dominant_angles_array = dominant_angles_array

        # Rescale the range of the pixels to have discrete grey level values
        greylevels = self.greylevels

        new_max = greylevels - 1
        new_min = 0

        dominant_angles_max = dominant_angles_array.max()
        dominant_angles_min = dominant_angles_array.min()

        dominant_angles_shaped = (dominant_angles_array - dominant_angles_min) / (
                    dominant_angles_max - dominant_angles_min)
        dominant_angles_shaped = dominant_angles_shaped * (new_max - new_min) + new_min
        dominant_angles_shaped = np.round(dominant_angles_shaped)
        dominant_angles_shaped = dominant_angles_shaped.astype(int)
        self.dominant_angles_shaped = dominant_angles_shaped

        haralick_features = np.empty((mask_height, mask_width, 13))
        full_images = []
        full_masked_images = []

        number_of_features = len(self.haralick_feature_list)

        haralick_feature_list = self.haralick_feature_list
        if haralick_feature_list[0] == HaralickFeature.All:
            number_of_features = 13
        for feature in range(number_of_features):

            if number_of_features != 13:
                feature = haralick_feature_list.pop().value
            if self.verbose_logging:
                print(f'Calculating feature {feature + 1}:')
            haralick_features[:, :, feature] = self.get_haralick_feature(dominant_angles_shaped, feature, greylevels,
                                                                            self.haralick_window_size, symmetric=False,
                                                                            mean=True)

            single_feature = scale_array_for_image(haralick_features[:, :, feature].astype('float64'))

            # This is the patch overlayed on top of the whole image.
            full_image = img_array
            full_image = scale_array_for_image(full_image)
            full_image[mask_min_y:mask_max_y, mask_min_x:mask_max_x] = single_feature
            full_images.append(full_image)

            # This is an array the same shape as the image but with zeros for the non-patch values.
            full_masked_image = np.zeros(img_array.shape)
            full_masked_image[mask_min_y:mask_max_y, mask_min_x:mask_max_x] = single_feature
            full_masked_images.append(full_masked_image)

            if self.verbose_logging:
                print(f'Calculated feature {feature + 1}.')

        self.haralick_features = haralick_features
        self.full_images = full_images
        self.full_masked_images = full_masked_images

        return full_masked_images
