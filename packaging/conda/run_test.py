import numpy as np
from collageradiomics import Collage

for shape in ((8, 9), (8, 9, 3)):
    image = np.random.default_rng(42).random(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[0:2, 1:3] = True
    output = Collage(image, mask, svd_radius=1, num_unique_angles=4).execute()
    expected = shape + ((13,) if image.ndim == 2 else (13, 2))
    assert output.shape == expected
    assert np.isfinite(output[mask]).all()
    assert np.isnan(output[~mask]).all()
