clear all;
clear haralickfun
clear haralick2mex
mex haralick2mex.cpp
haralickfun=@haralick2mex

% load image
raw_image = imread('../../sample_data/ImageSlice.png');
raw_image = raw_image(:,:,1);
mask = zeros(size(raw_image));

% prepare patch
svd_radius = 5
patch_window_width = 30
patch_window_height = 30
mask_min_x = 252
mask_min_y = 193
mask_min_x = mask_min_x +1
mask_min_y = mask_min_y +1
mask_max_x = mask_min_x + patch_window_width - 1
mask_max_y = mask_min_y + patch_window_height - 1

mask(mask_min_y:mask_max_y, mask_min_x:mask_max_x) = 1;
cropped_array = raw_image(mask_min_y:mask_max_y, mask_min_x:mask_max_x, :);

% show patch
close all;
figure;
subplot(1,2,1);
imagesc(raw_image);
colormap gray;
hold on;
rectangle('Position', ...
  [mask_min_x, mask_min_y, patch_window_width, patch_window_height], ...
  'EdgeColor', 'cyan',
  'LineWidth', 3);
subplot(1,2,2);
imagesc(cropped_array);
colormap gray;
colorbar

% calculate collage
haralick_number = 13
[collage_map, volfeats, Gx, Gy, dominant_orientation_roi, volN]=compute_CoLlAGe2D(raw_image, mask, svd_radius, haralick_number);

% display gradients
figure;
subplot(1,2,1);
imagesc(Gx);
colormap jet; colorbar;
title('Gx');
subplot(1,2,2);
imagesc(Gy);
colormap jet; colorbar;
title('Gy');

% display dominant angles
figure;
imagesc(dominant_orientation_roi);
colormap jet; colorbar;
title('Dominant Angles (SVD)');

% display vol features
figure
imagesc(volN)
title('Color-binned Angles')
colormap jet; colorbar;

% manually test haralick
figure;
for i = 1:13
  subplot(3,5,i);
  imagesc(volfeats(:,:,i));
  colormap jet; colorbar;
  title(['Har' num2str(i)])
endfor
