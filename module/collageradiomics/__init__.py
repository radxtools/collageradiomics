"""CoLlAGe descriptors for finite, aligned 2D and 3D NumPy arrays."""

import logging
from collections.abc import Iterable
from enum import Enum, IntEnum
from typing import Any

import numpy as np
from _collageradiomics.orientation import dominant_angle as _svd_dominant_angle  # noqa: F401
from _collageradiomics.orientation import dominant_angles as _svd_dominant_angles
from _collageradiomics.progress import ProgressCallback, ProgressReporter
from _collageradiomics.texture import named_haralick
from _collageradiomics.validation import angles, integer, validate_arrays
from numpy.typing import NDArray
from skimage.feature import graycomatrix

Array = NDArray[Any]
FloatArray = NDArray[np.float64]
logger = logging.getLogger("collageradiomics")


class HaralickFeature(IntEnum):
    """Enumeration Helper For Haralick Features

    :param IntEnum: Enumeration Helper For Haralick Features
    :type IntEnum: HaralickFeature
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


class DifferenceVarianceInterpretation(Enum):
    """Feature 10 has two interpretations, as the variance of |x-y|
    or as the variance of P(|x-y|).
    See: https://ieeexplore.ieee.org/document/4309314

    :param Enum: Enumeration Helper For Haralick Features
    :type Enum: DifferenceVarianceInterpretation
    """

    XMinusYVariance = 0
    ProbabilityXMinusYVariance = 1


class Collage:
    """This is the main object in the Collage calculation system. Usage: create a Collage object and then call the :py:meth:`execute` function.

    :param image_array: image to run collage upon
    :type image_array: numpy.ndarray
    :param mask_array: mask that correlates with the image
    :type mask_array: numpy.ndarray
    :param svd_radius: radius of svd. Defaults to 5.
    :type svd_radius: int, optional
    :param verbose_logging: This parameter is now ignored. Please use the python logging module.
    :type verbose_logging: bool, optional
    :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [x*numpy.pi/4 for x in range(8)]
    :type cooccurence_angles: list, optional
    :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
    :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
    :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
    :type haralick_window_size: int, optional
    :param num_unique_angles: number of bins to use for the texture calculation. Defaults to 64.
    :type num_unique_angles: int, optional
    """

    _collage_output: FloatArray | None

    @property
    def img_array(self) -> Array:
        """
        The original image.

        :getter: Returns the original image array.
        :setter: Sets the original image array.
        :type: np.ndarray
        """
        return self._img_array

    @property
    def mask_array(self) -> NDArray[np.bool_]:
        """
        Array passed into Collage.

        :getter: Returns the original mask array.
        :setter: Sets the original mask array.
        :type: np.ndarray
        """
        return self._mask_array

    @property
    def is_3D(self) -> bool:
        """
        Whether we are using 3D collage calculations (True) or 2D (False)
        """
        return self._is_3D

    @property
    def svd_radius(self) -> int:
        """
        SVD radius is used to calculate the pixel radius
        for the dominant angle calculation.

        :getter: Returns the SVD radius.
        :setter: Sets the SVD radius.
        :type: int
        """
        return self._svd_radius

    @property
    def verbose_logging(self) -> bool:
        """
        This parameter is now ignored. Please use the python logging module.

        :getter: Returns True if on.
        :setter: Turns verbose logging off or on.
        :type: bool
        """
        return self._verbose_logging

    @property
    def cooccurence_angles(self) -> tuple[float, ...]:
        """
        Iterable of angles that will be used in the cooccurence matrix.

        :getter: Returns the Iterable of cooccurence angles.
        :setter: Sets the angles to be used in the cooccurence matrix.
        :type: int
        """
        return self._cooccurence_angles

    @property
    def difference_variance_interpretation(self) -> DifferenceVarianceInterpretation:
        """
        Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].
        Defaults to DifferenceVarianceInterpretation.XMinusYVariance.

        :getter: Returns requested variance interpretation.
        :setter: Sets requested variance interpretation.
        :type: DifferenceVarianceInterpretation
        """
        return self._difference_variance_interpretation

    @property
    def haralick_window_size(self) -> int:
        """
        Number of pixels around each pixel to calculate a haralick texture.

        :getter: Returns requested number of pixels.
        :setter: Sets requested number of pixels.
        :type: int
        """
        return self._haralick_window_size

    @property
    def num_unique_angles(self) -> int:
        """
        Number of bins to use for texture calculations. Defaults to 64.

        :getter: Returns requested number of unique angles to bin into.
        :type: int
        """
        return self._num_unique_angles

    @property
    def collage_output(self) -> FloatArray:
        """
        Array representing collage upon the mask within the full images.
        If the input was 2D, the output will be height×width×13 where "13" is the number of haralick textures.
        If the input was 3D, the output will be height×width×depth×13x2 where "2" is the primary angle (element 0) or the secondary angle (element 1)

        The output will have numpy.nan values everywhere outside the masked region.

        :getter: Returns array the same shape as the original image with collage in the mask region.
        :type: numpy.ndarray
        """
        if self._collage_output is None:
            raise RuntimeError("execute() must be called before requesting features")
        return self._collage_output

    @collage_output.setter
    def collage_output(self, value: FloatArray) -> None:
        self._collage_output = value

    def get_single_feature_output(self, which_feature: int | HaralickFeature) -> FloatArray:
        """
        Output a single collage output feature.
        If this was a 3D calculation, the output will be of size height×width×depth×2
        where the "2" represents the collage calculation from the primary angle (0) or secondary angle (1).

          :param which_feature: Either an integer from 0 to 12 (inclusive) or a HaralickFeature enum value
          :type which_feature HaralickFeature
        """
        if self._collage_output is None:
            raise RuntimeError("execute() must be called before requesting features")
        which_feature = integer(which_feature, "which_feature", minimum=0)
        if which_feature >= len(HaralickFeature):
            raise ValueError("which_feature must be between 0 and 12")
        if self.is_3D:
            return self.collage_output[:, :, :, which_feature, :]
        else:
            return self.collage_output[:, :, which_feature]

    def __init__(
        self,
        img_array: Array,
        mask_array: Array,
        svd_radius: int = 5,
        verbose_logging: bool = False,
        cooccurence_angles: Iterable[float] | None = None,
        difference_variance_interpretation: DifferenceVarianceInterpretation = DifferenceVarianceInterpretation.XMinusYVariance,
        haralick_window_size: int = -1,
        num_unique_angles: int = 64,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Designated initializer for Collage

        :param image_array: image to run collage upon
        :type image_array: numpy.ndarray
        :param mask_array: mask that correlates with the image
        :type mask_array: numpy.ndarray
        :param svd_radius: radius of svd. Defaults to 5.
        :type svd_radius: int, optional
        :param verbose_logging: This parameter is now ignored. Please use the python logging module.
        :type verbose_logging: bool, optional
        :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [x * np.pi/4 for x in range(8)]
        :type cooccurence_angles: list, optional
        :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
        :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
        :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
        :type haralick_window_size: int, optional
        :param num_unique_angles: number of bins to use for the texture calculation. Defaults to 64.
        :type num_unique_angles: int, optional
        """

        self._collage_output = None
        self._progress = ProgressReporter(progress_callback)
        self._texture_stage = "haralick"
        logger.debug("Collage Module Initialized")

        validate_arrays(img_array, mask_array)
        svd_radius = integer(svd_radius, "SVD radius")
        num_unique_angles = integer(num_unique_angles, "num_unique_angles", minimum=2)
        cooccurence_angles = angles(cooccurence_angles)
        if not isinstance(difference_variance_interpretation, DifferenceVarianceInterpretation):
            raise TypeError(
                "difference_variance_interpretation must be a DifferenceVarianceInterpretation"
            )

        if isinstance(haralick_window_size, (int, np.integer)) and haralick_window_size == -1:
            self._haralick_window_size = svd_radius * 2 + 1
        else:
            self._haralick_window_size = integer(
                haralick_window_size, "haralick_window_size", minimum=3
            )
        if self._haralick_window_size % 2 == 0:
            raise ValueError("haralick_window_size must be odd")

        self._is_3D = img_array.ndim == 3
        logger.debug(f"Running 3D Collage = {self.is_3D}")

        self._img_array = img_array
        if not self.is_3D:
            # in the case of a single 2D slice, give it a third dimension of unit length
            self._img_array = self._img_array.reshape(self._img_array.shape + (1,))

        # threshold mask
        uniqueValues = np.unique(mask_array)
        numberOfValues = len(uniqueValues)
        if numberOfValues > 2:
            logger.info(
                f"Warning: Mask is not binary. Considering all {numberOfValues} nonzero values in the mask as a value of True."
            )
        thresholded_mask_array = mask_array != 0

        # make correct shape
        thresholded_mask_array = thresholded_mask_array.reshape(self.img_array.shape)

        # extract rectangular area of mask
        non_zero_indices = np.argwhere(thresholded_mask_array)
        min_mask_coordinates = non_zero_indices.min(0)
        max_mask_coordinates = non_zero_indices.max(0) + 1
        self.mask_min_x = min_mask_coordinates[1]
        self.mask_min_y = min_mask_coordinates[0]
        self.mask_min_z = min_mask_coordinates[2]
        self.mask_max_x = max_mask_coordinates[1]
        self.mask_max_y = max_mask_coordinates[0]
        self.mask_max_z = max_mask_coordinates[2]

        cropped_mask_array = thresholded_mask_array[
            self.mask_min_y : self.mask_max_y,
            self.mask_min_x : self.mask_max_x,
            self.mask_min_z : self.mask_max_z,
        ]

        # store variables internally
        self._mask_array = cropped_mask_array

        self._svd_radius = svd_radius
        self._verbose_logging = verbose_logging

        self._cooccurence_angles = cooccurence_angles
        self._difference_variance_interpretation = difference_variance_interpretation

        self._num_unique_angles = num_unique_angles

    def _calculate_haralick_feature_values(
        self, img_array: Array, center_x: int, center_y: int
    ) -> FloatArray:
        """Calculate the thirteen named features in a centered, odd-sized window."""
        half = self.haralick_window_size // 2
        patch = img_array[
            max(0, center_y - half) : min(img_array.shape[0], center_y + half + 1),
            max(0, center_x - half) : min(img_array.shape[1], center_x + half + 1),
        ]
        matrix = graycomatrix(patch, [1], self.cooccurence_angles, levels=self.num_unique_angles)
        matrix = np.sum(matrix[:, :, 0, :], axis=2)
        return named_haralick(
            matrix,
            self.difference_variance_interpretation
            == DifferenceVarianceInterpretation.XMinusYVariance,
        )

    def _calculate_haralick_textures(self, dominant_angles: Array) -> FloatArray:
        """Calculate masked voxels using a halo of unmasked neighboring angles."""
        low, high = float(dominant_angles.min()), float(dominant_angles.max())
        if high - low <= np.finfo(float).eps:
            binned = np.zeros(dominant_angles.shape, dtype=np.int32)
        else:
            scaled = (dominant_angles.astype(float) - low) / (high - low)
            binned = np.rint(scaled * (self.num_unique_angles - 1)).astype(np.int32)
        output = np.full(self.mask_array.shape + (13,), np.nan)
        halo = self.haralick_window_size // 2
        coordinates = np.argwhere(self.mask_array)
        total = len(coordinates)
        self._progress(self._texture_stage, 0, total)
        for completed, (y, x, z) in enumerate(coordinates, 1):
            output[y, x, z] = self._calculate_haralick_feature_values(
                binned[:, :, z],
                int(x) + halo,
                int(y) + halo,
            )
            self._progress(self._texture_stage, completed, total)
        return output

    def execute(self) -> FloatArray:
        """Return feature maps at input size, with NaN only outside the mask.

        Image edges are extended by replication. SVD neighborhoods and Haralick
        neighborhoods both include their complete halo, including for small ROIs.
        In 3D, gradients use neighboring slices and textures remain slice-wise.
        """
        self._collage_output = None
        half = self.haralick_window_size // 2
        halo = self.svd_radius + half
        image = self.img_array.astype(np.float64, copy=False)
        y0, y1 = int(self.mask_min_y), int(self.mask_max_y)
        x0, x1 = int(self.mask_min_x), int(self.mask_max_x)
        z0, z1 = int(self.mask_min_z), int(self.mask_max_z)
        height, width, depth = y1 - y0, x1 - x0, z1 - z0
        cy0, cy1 = max(0, y0 - halo), min(image.shape[0], y1 + halo)
        cx0, cx1 = max(0, x0 - halo), min(image.shape[1], x1 + halo)
        cz0, cz1 = max(0, z0 - 1), min(image.shape[2], z1 + 1)
        patch = image[cy0:cy1, cx0:cx1, cz0:cz1]
        patch = np.pad(
            patch,
            (
                (max(0, halo - y0), max(0, y1 + halo - image.shape[0])),
                (max(0, halo - x0), max(0, x1 + halo - image.shape[1])),
                (0, 0),
            ),
            mode="edge",
        )
        magnitude = np.max(np.abs(patch))
        if magnitude > 1:
            patch = patch / magnitude
        z_slice = slice(z0 - cz0, z0 - cz0 + depth)
        dx = np.gradient(patch, axis=1)[:, :, z_slice]
        dy = np.gradient(patch, axis=0)[:, :, z_slice]
        dz = (np.gradient(patch, axis=2) if patch.shape[2] > 1 else np.zeros_like(patch))[
            :, :, z_slice
        ]
        self.dx, self.dy, self.dz = dx, dy, dz
        dominant = _svd_dominant_angles(
            dx,
            dy,
            dz,
            self.svd_radius,
            is_3d=self.is_3D,
            progress=lambda done, total: self._progress("orientation", done, total),
        )
        self.dominant_angles = dominant[half : half + height, half : half + width]
        num_angles = 2 if self.is_3D else 1
        features = np.empty((height, width, depth, 13, num_angles))
        for angle in range(num_angles):
            self._texture_stage = "haralick-theta" if angle == 0 else "haralick-phi"
            features[:, :, :, :, angle] = self._calculate_haralick_textures(
                dominant[:, :, :, angle]
            )
        output = np.full(image.shape + (13, num_angles), np.nan)
        output[y0:y1, x0:x1, z0:z1] = features
        if not self.is_3D:
            output = output[:, :, 0, :, 0]
        self.collage_output = output
        return output
