# Overview
This contains code from ccipd to use as a benchmark/baseline for expected collage results.

# Usage
## Matlab/Octave

Load:
```
cd Baseline/matlab;
raw_image = imread('../../sample_data/ImageSlice.png');
mask = imread('../../sample_data/ImageMask.png');
mask = mask(:,:,1);
```

Calculate:
```
collage = compute_CoLlAGe2D(raw_image, mask, 5, 6);
```

Display:
```
close all;

figure; imshow(raw_image);
title('Image');
axis on;

figure; imagesc(mask);
title('Mask');

figure;
imagesc(collage(:,:,end));
colorbar;
title('Collage');
```
