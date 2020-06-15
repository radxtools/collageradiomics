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
    dx_patch = dx_windows[y, x]
    dy_patch = dy_windows[y, x]
    
    window_area = dx_patch.size
    flattened_gradients = np.zeros((window_area, 2))
    flattened_gradients[:,0] = np.reshape(dx_patch, ((window_area)), order='F')
    flattened_gradients[:,1] = np.reshape(dy_patch, ((window_area)), order='F')
    
    _, _, V = linalg.svd(flattened_gradients)
    dominant_angle = math.atan2(V[0, 0], V[1, 0])
    
    return dominant_angle

def show_colored_image(figure, axis, image_data, colormap=plt.cm.jet):
    image = axis.imshow(image_data, cmap=colormap)
    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(image, cax=colorbar_axis)

def create_highlighted_rectangle(x, y, w, h):
    return Rectangle((x, y), w, h, linewidth=3, edgecolor='cyan', facecolor='none')

def highlight_rectangle_on_image(image_data, min_x, min_y, w, h, colormap=plt.cm.gray):
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
    print(iterable.shape)
    min_x, min_y = np.min(iterable[0], axis=0)
    max_x, max_y = np.max(iterable[0], axis=0)
    return np.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

def bbox1(img):
    a = np.where(img == 255)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

class Collage:
    def __init__(self, img_array, mask_array, svd_radius, verbose_logging=False):
        self.img_array = img_array
        if len(mask_array.shape) > 2:
            mask_array = mask_array[:,:,0]
        trimmed_mask_array = (mask_array == 255).astype(int)
        non_zero_indices = np.argwhere(trimmed_mask_array)
        (min_y, min_x), (max_y, max_x) = non_zero_indices.min(0), non_zero_indices.max(0) + 1 
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

    @classmethod
    def from_rectangle(cls, img_array, mask_min_x, mask_min_y, patch_window_width, patch_window_height, svd_radius, verbose_logging=False):
        mask_array = np.zeros((img_array.shape[0], img_array.shape[1]))
        mask_array[mask_min_y:mask_min_y + patch_window_height, mask_min_x:mask_min_x + patch_window_width] = 255
        return cls(img_array, mask_array, svd_radius, verbose_logging)

    def get_haralick_mt_value(self, img_array, center_x, center_y, window_size, greylevels, haralick_feature, symmetric, mean):
        # extract subpart of image (todo: pass in result from view_as_windows)
        min_x = int(max(0, center_x - window_size / 2 - 1))
        min_y = int(max(0, center_y - window_size / 2 - 1))
        max_x = int(min(img_array.shape[1] - 1, center_x + window_size / 2 + 1))
        max_y = int(min(img_array.shape[0] - 1, center_y + window_size / 2 + 1))
        cropped_img_array = img_array[min_y:max_y, min_x:max_x]
        
        # co-occurence matrix of all 8 directions and sum them
        cmat = greycomatrix(cropped_img_array, [1], [0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4], levels=greylevels)
        cmat = np.sum(cmat, axis=3)
        cmat = cmat[:,:,0]
        
        # extract haralick using mahotas library:
        har_feature = mt.features.texture.haralick_features([cmat], return_mean=mean)
        
        # output:
        if mean:
            return har_feature[haralick_feature]
        # direction is undefined -- do we need this?
        direction = 0
        return har_feature[direction, haralick_feature]

    def get_haralick_mt_feature(self, img, desired_haralick_feature, greylevels, haralick_window_size, symmetric=False, mean=False):
        haralick_image = np.zeros(img.shape)
        h, w = img.shape
        for pos in product(range(w), range(h)):
            if self.mask_array[pos[1]][pos[0]] != 0: 
                result = self.get_haralick_mt_value(img, pos[0], pos[1], haralick_window_size, greylevels, desired_haralick_feature, symmetric, mean)
                haralick_image[pos[1], pos[0]] = result
        return haralick_image

    def execute(self):
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

        # unused -- do we still need this?
        # haralick_radius = svd_radius * 2 + 1

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
                if (random.randint(0,500)==0):
                    if self.verbose_logging:
                        print(f'x={current_svd_center_x}, y={current_svd_center_y}')
                        print(f'angle={current_dominant_angle}')

        if self.verbose_logging:
            print('Done calculating dominant angles.')

        self.dominant_angles_array = dominant_angles_array

        # Rescale the range of the pixels to have discrete grey level values
        greylevels = 64

        new_max = greylevels-1
        new_min = 0

        dominant_angles_max = dominant_angles_array.max()
        dominant_angles_min = dominant_angles_array.min()

        dominant_angles_shaped = (dominant_angles_array - dominant_angles_min) / (dominant_angles_max - dominant_angles_min)
        dominant_angles_shaped = dominant_angles_shaped * (new_max - new_min) + new_min
        dominant_angles_shaped = np.round(dominant_angles_shaped)
        dominant_angles_shaped = dominant_angles_shaped.astype(int)
        self.dominant_angles_shaped =  dominant_angles_shaped

        # Calculate haralick
        haralick_window_size = svd_radius * 2 + 1

        haralick_features = np.empty((patch_window_height, patch_window_width, 13))

        for feature in range(1):
            if self.verbose_logging:
                print(f'Calculating feature {feature+1}:')
            haralick_features[:,:,feature] = self.get_haralick_mt_feature(dominant_angles_shaped, feature, greylevels, haralick_window_size, symmetric=False, mean=True)
            if self.verbose_logging:
                print(f'Calculated feature {feature+1}.')

        self.haralick_features = haralick_features

        return haralick_features
