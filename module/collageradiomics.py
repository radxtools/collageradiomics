import matplotlib.pyplot as plt
import numpy as np
import random
import math
import mahotas as mt

from matplotlib.patches import Rectangle
from scipy import linalg
from skimage.util.shape import view_as_windows
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product
from skimage.feature.texture import greycomatrix

def svd_dominant_angle(x, y, dx_windows, dy_windows):
    """Calculates the dominate angle at the coordinate within the windows.

    Parameters
    ----------
        x : int
            x value of coordinate
        y : int
            y value of coordinate
        dx_windows : numpy.array
            dx windows of x, y shape to run svd upon
        dy_windows : numpy.array
            dy windows of x, y shape to run svd upon

    Returns
    -------
        float 
            dominant angle at x, y
    """
    dx_patch = dx_windows[y, x]
    dy_patch = dy_windows[y, x]
    
    window_area = dx_patch.size
    flattened_gradients = np.zeros((window_area, 2))
    flattened_gradients[:,0] = np.reshape(dx_patch, ((window_area)), order='F')
    flattened_gradients[:,1] = np.reshape(dy_patch, ((window_area)), order='F')
    
    _, _, V = linalg.svd(flattened_gradients)
    dominant_angle = math.atan2(V[0, 0], V[1, 0
    
    return dominant_angle

def show_colored_image(figure, axis, image_data, colormap=plt.cm.jet):
    """Helper method to show a colored image in matplotlib.

    Parameters
    ----------
        figure : matplotlib.pyplot.figure
            figure upon which to display
        axis : matplotlib.pyplot.axis
            axis upon which to display
        image_data : numpy.array
            image to display
        colormap : matplotlib.colors.Colormap, optional
            color map to convert for display. Defaults to plt.cm.jet.
    """
    image = axis.imshow(image_data, cmap=colormap)
    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(image, cax=colorbar_axis)

def create_highlighted_rectangle(x, y, w, h):
    """Creates a matplotlib Rectangle object for a highlight effect

    Parameters
    ----------
        x : int
            x location to start rectangle
        y : int
            y location to start rectangle
        w : int
            width of rectangle
        h : int
            height of rectangle

    Returns
    -------
        matplotlib.patches.Rectangle
            Rectangle used to highlight within a plot
    """
    return Rectangle((x, y), w, h, linewidth=3, edgecolor='cyan', facecolor='none')

def highlight_rectangle_on_image(image_data, min_x, min_y, w, h, colormap=plt.cm.gray):
    """Highlights a rectangle on an image at the passed in coordinate.

    Parameters
    ----------
        image_data : numpy.array
            image to highlight
        min_x : int
            x location to start highlight
        min_y : int
            y location to start highlight
        w : int
            width of highlight rectangle
        h : int
            height of highlight rectangle
        colormap : matplotlib.colors.Colormap, optional
            color map to convert for display. Defaults to plt.cm.jet.

    Returns
    -------
        numpy.array
            image array with highlighted rectangle
    """
    figure, axes = plt.subplots(1,2, figsize=(15,15))

    # Highlight window within image.
    show_colored_image(figure, axes[0], image_data, colormap)
    axes[0].add_patch(create_highlighted_rectangle(min_x, min_y, w, h))

    # Crop window.
    cropped_array = image_data[min_y:min_y+h, min_x:min_x+w]
    axes[1].set_title(f'Cropped Region ({w}x{h})')
    show_colored_image(figure, axes[1], cropped_array, colormap)
    
    plt.show()
    
    return cropped_array

def bounding_box(iterable):
    """Creates a bounding box within an iterable

    Parameters
    ----------
        iterable : iterable type
            array/list/iterable type to grab a boundingbox

    Returns
    -------
        numpy.array
            array representing region of bounding box
    """
    print(iterable.shape)
    min_x, min_y = np.min(iterable[0], axis=0)
    max_x, max_y = np.max(iterable[0], axis=0)
    return np.array(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)

def bbox1(img):
    """Calculates bounding box where pixel values are 255

    Parameters
    ----------
        img : numpy.array
            image to find bounding box

    Returns
    -------
        tuple (x_min, x_max, y_min, y_max)
            tuple representing bounding box minx, maxx, miny, maxy
    """
    a = np.where(img == 255)
    bbox = np.min(a[0, np.max(a[0, np.min(a[1, np.max(a[1
    return bbox

def scale_array_for_image(array_to_scale):
    """Scales an array from 0-255 integer values

    Parameters
    ----------
        array_to_scale : numpy.array
            array to scale

    Returns
    -------
        numpy.array
            array scaled from 0-255
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

    Parameters
    ----------
        Enum : HaralickFeature
            Enumeration Helper For Haralick Features
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

    Parameters
    ----------
        Enum DifferenceVarianceInterpretation
            Interpretations of Feature 10
    """
    XMinusYVariance = 0
    ProbabilityXMinusYVariance = 1

class CollageCollection:
    """Enables computation on multiple collage images at the same time.
    """
    def __init__(self,
    images_array, 
    masks_array,
    svd_radius=5, 
    verbose_logging=False,
    haralick_feature_list=[HaralickFeature.All], 
    log_sample_rate=500, 
    cooccurence_angles=[0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4],
    difference_variance_interpretation = DifferenceVarianceInterpretation.XMinusYVariance,
    haralick_window_size=-1,
    greylevels = 64,
    ):
        """Designated initializer for CollageCollection

        Parameters
        ----------
            images_array : numpy.array
                array of images to run collage upon
            masks_array : numpy.array
                array of masks to correspond with the images
            svd_radius : int, optional
                radius of svd. Defaults to 5.
            verbose_logging : bool, optional
                turning this on will log intermediate results]. Defaults to False.
            haralick_feature_list : [HaralickFeature], optional
                array of features to calculate. Defaults to [HaralickFeature.All].
            log_sample_rate : int, optional
                higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
            cooccurence_angles : list, optional
                list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
            difference_variance_interpretation : DifferenceVarianceInterpretation, optional
                Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).]. Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            haralick_window_size : int, optional
                size of rolling window for texture calculations. Defaults to -1.
            greylevels : int, optional
                number of bins to use for the grey levels. Defaults to 64.
        """
        self.images_array = images_array
        self.masks_array =  masks_array
        self.svd_radius = svd_radius
        self.verbose_logging = verbose_logging
        self.haralick_feature_list = haralick_feature_list
        self.log_sample_rate = log_sample_rate
        self.cooccurence_angles = cooccurence_angles
        self.difference_variance_interpretation = difference_variance_interpretation
        self.haralick_window_size = haralick_window_size
        self.greylevels = greylevels
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
    """Calculates the collageradiomics algorithm.
    """
    def __init__(self, 
    img_array, 
    mask_array, 
    svd_radius=5, 
    verbose_logging=False, 
    haralick_feature_list=[HaralickFeature.All], 
    log_sample_rate=500, 
    cooccurence_angles=[0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4],
    difference_variance_interpretation = DifferenceVarianceInterpretation.XMinusYVariance,
    haralick_window_size=-1,
    greylevels = 64,
    ):
        """Designated initializer for Collage

        Parameters
        ----------
            image_array : numpy.array
                image to run collage upon
            mask_array : numpy.array
                mask that correlates with the image
            svd_radius : int, optional
                radius of svd. Defaults to 5.
            verbose_logging : bool, optional
                turning this on will log intermediate results]. Defaults to False.
            haralick_feature_list : [HaralickFeature], optional
                array of features to calculate. Defaults to [HaralickFeature.All].
            log_sample_rate : int, optional
                higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
            cooccurence_angles : list, optional
                list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
            difference_variance_interpretation : DifferenceVarianceInterpretation, optional
                Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).]. Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            haralick_window_size : int, optional
                size of rolling window for texture calculations. Defaults to -1.
            greylevels : int, optional
                number of bins to use for the grey levels. Defaults to 64.
        """
        if haralick_window_size == -1:
            self.haralick_window_size = svd_radius * 2 + 1
        else:
            self.haralick_window_size = haralick_window_size

        if self.haralick_window_size < 1:
            raise Exception('Haralick windows size must be at least 1 pixel.')

        if svd_radius < 1:
            raise Exception('SVD radius must be at least 1 pixel')

        if greylevels < 1:
            raise Exception('greylevels must contain at least 1 bin')

        if mask_array.shape[0] != img_array.shape[0] or mask_array.shape[1] != img_array.shape[1]:
            raise Exception('Mask must be same size as image.')

        self.img_array = img_array
        
        if self.img_array.shape[0] < self.haralick_window_size or self.img_array.shape[1] < self.haralick_window_size:
            raise Exception(f'Image must be at least {self.haralick_window_size}x{self.haralick_window_size} to create appropriate windows.')

        if len(mask_array.shape) > 2:
            mask_array = mask_array[:,:,0]
        trimmed_mask_array = (mask_array == 255).astype('float64')
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
        self.patch_window_width = self.mask_max_x - self.mask_min_x
        self.patch_window_height = self.mask_max_y - self.mask_min_y
        self.svd_radius = svd_radius
        self.verbose_logging = verbose_logging
        self.mask_array = scaled_mask_array

        self.haralick_feature_list = haralick_feature_list
        self.feature_count = len(haralick_feature_list)
        self.log_sample_rate = log_sample_rate
        self.cooccurence_angles = cooccurence_angles
        self.difference_variance_interpretation = difference_variance_interpretation

        self.greylevels = greylevels
        self.haralick_features = []

    @classmethod
    def from_rectangle(cls, 
    img_array, 
    mask_min_x, 
    mask_min_y, 
    patch_window_width, 
    patch_window_height, 
    svd_radius=5, 
    verbose_logging=False,
    haralick_feature_list=[HaralickFeature.All], 
    log_sample_rate=500, 
    cooccurence_angles=[0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4],
    difference_variance_interpretation = DifferenceVarianceInterpretation.XMinusYVariance,
    haralick_window_size=-1,
    greylevels = 64,
    ):
        """Calculates collageradiomics within the mask coordinates

        Parameters
        ----------
            img_array : numpy.array
                image to run collage upon
            mask_min_x : int
                x location of window
            mask_min_y : int
                y location of window
            patch_window_width : int
                window width
            patch_window_height : int
                window height
            svd_radius : int, optional
                radius of svd. Defaults to 5.
            verbose_logging : bool, optional
                turning this on will log intermediate results]. Defaults to False.
            haralick_feature_list : [HaralickFeature], optional
                array of features to calculate. Defaults to [HaralickFeature.All].
            log_sample_rate : int, optional
                higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
            cooccurence_angles : list, optional
                list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
            difference_variance_interpretation : DifferenceVarianceInterpretation, optional
                Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).]. Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            haralick_window_size : int, optional
                size of rolling window for texture calculations. Defaults to -1.
            greylevels : int, optional
                number of bins to use for the grey levels. Defaults to 64.

        Returns
        -------
            Collage
                Collage object to run collage on a rectangular section of the image]
        """
        mask_array = np.zeros((img_array.shape[0], img_array.shape[1)
        mask_array[mask_min_y:mask_min_y + patch_window_height, mask_min_x:mask_min_x + patch_window_width] = 255
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
    cooccurence_angles=[0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4],
    difference_variance_interpretation = DifferenceVarianceInterpretation.XMinusYVariance,
    haralick_window_size=-1,
    greylevels = 64,
    ):
        """Helper method to run collage on multiple images.

        Parameters
        ----------
            image_array : numpy.array
                image to run collage upon
            mask_array : numpy.array
                mask that correlates with the image
            svd_radius : int, optional
                radius of svd. Defaults to 5.
            verbose_logging : bool, optional
                turning this on will log intermediate results]. Defaults to False.
            haralick_feature_list : [HaralickFeature], optional
                array of features to calculate. Defaults to [HaralickFeature.All].
            log_sample_rate : int, optional
                higher values will log more svd angles, this only works with verbose logging. Defaults to 500.
            cooccurence_angles : list, optional
                list of angles to use in the cooccurence matrix. Defaults to [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4].
            difference_variance_interpretation : DifferenceVarianceInterpretation, optional
                Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).]. Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            haralick_window_size : int, optional
                size of rolling window for texture calculations. Defaults to -1.
            greylevels : int, optional
                number of bins to use for the grey levels. Defaults to 64.

        Returns
        -------
            CollageCollection
                CollageCollection object that works similarly to Collage. Call execute() to run on all images]
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
    

    def get_haralick_mt_value(self, img_array, center_x, center_y, window_size, greylevels, haralick_feature, symmetric, mean):
        """Gets the haralick texture value at the center of an x, y coordinate.

        Parameters
        ----------
            image_array : numpy.array
                image to calculate texture
            center_x : int
                x center of coordinate
            center_y int
                y center of coordinate
            window_size : int
                size of window to pull for calculation
            greylevels : int
                number of bins
            haralick_feature : HaralickFeature
                desired haralick feature
            symmetric : bool
                whether or not we should use the symmetrical cooccurence matrix
            mean : bool
                whether we return the mean of the feature or not

        Returns
        -------
            float
                number representing value of haralick texture at coordinate
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
        cooccurence_matrix = cooccurence_matrix[:,:,0]
        
        # extract haralick using mahotas library:
        har_feature = mt.features.texture.haralick_featurescooccurence_matrix], return_mean=mean)
        
        # output:
        if mean:
            return har_feature[haralick_feature]
        return har_feature[0, haralick_feature]

    def get_haralick_mt_feature(self, img, desired_haralick_feature, greylevels, haralick_window_size, symmetric=False, mean=False):
        """Gets haralick image within the mask

        Parameters
        ----------
            img : numpy.array
                image to get feature from
            desired_haralick_feature : Haralick Feature
                which feature to calculate
            greylevels : int
                number of bins
            haralick_window_size : int
                size of window around pixels to calculate haralick value
            symmetric (bool, optional)
                whether or not we should use the symmetrical cooccurence matrix. Defaults to False.
            mean (bool, optional)
                whether we return the mean of the feature or not. Defaults to False.

        Returns
        -------
            numpy.array
                image representing haralick texture
        """
        haralick_image = np.zeros(img.shape)
        h, w = img.shape
        for pos in product(range(w), range(h)):
            if self.mask_array[pos[1]][pos[0]] != 0: 
                result = self.get_haralick_mt_value(img, pos[0], pos[1], haralick_window_size, greylevels, desired_haralick_feature, symmetric, mean)
                haralick_image[pos[1], pos[0]] = result
        return haralick_image

    def execute(self):
        """Begins haralick calculation.

        Returns
        -------
            numpy.array
                image at original size that only has the masked section filled in with collage calculations]
        """
        mask_min_x = int(self.mask_min_x)
        mask_min_y = int(self.mask_min_y)
        mask_max_x = int(self.mask_max_x)
        mask_max_y = int(self.mask_max_y)
        patch_window_width = int(self.patch_window_width)
        patch_window_height = int(self.patch_window_height)
        svd_radius = self.svd_radius
        img_array = self.img_array[:,:,0]
        if self.verbose_logging:
            print(f'IMAGE:\nwidth={img_array.shape[1]} height={img_array.shape[0]}')

        cropped_array = img_array[mask_min_y:mask_min_y+patch_window_height, mask_min_x:mask_min_x+patch_window_width]
        if self.verbose_logging:
            print(f'Cropped Array Shape: {cropped_array.shape}')

        # Extend outwards
        padded_mask_min_x = max(mask_min_x - svd_radius, 0)
        padded_mask_min_y = max(mask_min_y - svd_radius, 0)
        padded_mask_max_x = min(mask_max_x + svd_radius, img_array.shape[1]-1)
        padded_mask_max_y = min(mask_max_y + svd_radius, img_array.shape[0]-1)
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

        dominant_angles_array = np.zeros((patch_window_height,patch_window_width), np.single)

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

        center_x_range = range(dx_windows.shape[1
        center_y_range = range(dx_windows.shape[0

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

        dominant_angles_shaped = (dominant_angles_array - dominant_angles_min) / (dominant_angles_max - dominant_angles_min)
        dominant_angles_shaped = dominant_angles_shaped * (new_max - new_min) + new_min
        dominant_angles_shaped = np.round(dominant_angles_shaped)
        dominant_angles_shaped = dominant_angles_shaped.astype(int)
        self.dominant_angles_shaped =  dominant_angles_shaped

        haralick_features = np.empty((patch_window_height, patch_window_width, 13))
        full_images = []
        full_masked_images = []

        number_of_features =  len(self.haralick_feature_list)

        haralick_feature_list = self.haralick_feature_list
        if self.haralick_feature_list[0] == HaralickFeature.All:
            number_of_features = 13
        for feature in range(number_of_features):

            if number_of_features != 13:
                feature = self.haralick_feature_list.pop().value
            if self.verbose_logging:
                print(f'Calculating feature {feature+1}:')
            haralick_features[:,:,feature] = self.get_haralick_mt_feature(dominant_angles_shaped, feature, greylevels, self.haralick_window_size, symmetric=False, mean=True)
            
            single_feature = scale_array_for_image(haralick_features[:,:,feature].astype('float64'))
            
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
                print(f'Calculated feature {feature+1}.')

        self.haralick_features = haralick_features
        self.haralick_feature_list = haralick_feature_list
        self.full_images = full_images
        self.full_masked_images = full_masked_images

        return full_masked_images
