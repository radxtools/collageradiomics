Installation and extraction
===========================

Use Python 3.10 or newer in a virtual environment. Until 0.4 is published, install
this checkout to use the corrected results and new CLI::

   python3 -m venv env_collage
   source env_collage/bin/activate
   python -m pip install -e './module[cli]'

For a published release, use ``python -m pip install 'collageradiomics[cli]'``.
The numerical core requires NumPy, SciPy, scikit-image, and Mahotas. Image I/O and
Click are optional CLI dependencies; plotting, DICOM, and notebooks are examples
extras. ``pip install -r requirements.txt`` installs the checkout with all examples.

Array API
---------

Images and masks must be finite NumPy arrays with matching shape. Every nonzero
mask label belongs to the ROI. Empty masks and invalid parameters raise a clear
``TypeError`` or ``ValueError``. Parameters use an integer SVD radius of at least
one, an odd Haralick window of at least three, and at least two angle bins.
The default window ``-1`` becomes ``2*svd_radius+1``.

.. code-block:: python

   import numpy as np
   from collageradiomics import Collage, HaralickFeature

   image = np.random.default_rng(42).random((32, 32))
   mask = np.zeros(image.shape, dtype=bool)
   mask[8:24, 8:24] = True
   collage = Collage(image, mask, svd_radius=2, num_unique_angles=8)
   maps = collage.execute()  # (32, 32, 13)
   entropy = collage.get_single_feature_output(HaralickFeature.Entropy)
   mean_entropy = entropy[mask].mean()

A 3D array uses ``(height,width,depth)`` and returns ``(H,W,D,13,2)``. The final
axis contains theta (0) and phi (1). Pixels outside the mask are NaN; filter them
before calculating summaries. ``np.save`` preserves numeric feature maps and NaNs.

The optional ``progress_callback(stage,completed,total)`` receives throttled
updates; raising an exception cancels extraction. Logging uses the
``collageradiomics`` logger. The legacy ``verbose_logging`` parameter is ignored.

Medical-image CLI
-----------------

Single images produce a FeatureName,Value CSV::

   collageradiomics -i image.nii.gz -m mask.nii.gz -o features.csv --extendstats

For batch extraction create a CSV with ``ID,Image,Mask`` columns::

   collageradiomics -i cases.csv -o features.csv --verbose

Paths in the CSV are relative to its directory. Each successful case produces a
row; failures appear in ``errors_features.csv`` and cause a nonzero exit code.
Mixed 2D/3D batches use a union of feature columns. Image and mask size, spacing,
origin, direction, and scalar dimensionality must match. Align mismatched masks
before extraction. ``--dimensions 2`` accepts 2D images or single-slice volumes;
it does not silently discard slices. ``--label`` selects a specific label.

``--haralickwindow``, ``--svdradius``, ``--binsize``, ``--difference-variance`` and
repeatable ``--cooccurrence-angle`` options expose the numerical settings.
``collageradiomicscli.py`` remains a compatibility command.

Examples and Slicer
-------------------

Install ``./module[examples]`` and open ``jupyter/examples/example.ipynb`` using
``jupyter lab``. It uses deterministic synthetic data and inline Matplotlib
figures; it never opens an external image editor.

``python module/test_script.py --output-dir output_data`` reads the bundled DICOM,
saves lossless ``collage_features.npy`` and thirteen derived DICOM previews.
Stored value zero is reserved for padding; rescale tags retain feature units.
``python examples/isbi.py IMAGE MASK --output-dir output_data`` accepts a local
IBSI pair and writes raw maps plus mean, population SD, skewness, and kurtosis.

The extension in ``slicer/collageradiomics`` requires Slicer 5.10 or newer. It
accepts labelmaps and visible segmentations, supports both angles and cancellation,
and preserves image geometry and parent transforms. Its README contains native
test instructions. CI runs the extension in a verified Slicer 5.12.4 bundle.
