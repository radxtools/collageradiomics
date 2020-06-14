% load image
raw_image = imread('../../sample_data/ImageSlice.png');
raw_image = raw_image(:,:,1);

% for random patch generation
svd_radius_range = [5, 10];
[raw_image_height, raw_image_width] = size(raw_image)
patch_window_width_range  = raw_image_width  * [0.05, 0.1];
patch_window_height_range = raw_image_height * [0.05, 0.1];
mask_min_x_range = raw_image_width  * [0.2, 0.8];
mask_min_y_range = raw_image_height * [0.2, 0.8];

total_seconds = 0;
N = 5
for i = 1:N
  
  % generate random parameters
  disp([num2str(i) '/' num2str(N)]);
  svd_radius = randi(int32(svd_radius_range))
  patch_window_width = randi(int32(patch_window_width_range))
  patch_window_height = randi(int32(patch_window_height_range))
  mask_min_x = randi(int32(mask_min_x_range));
  mask_min_y = randi(int32(mask_min_y_range));

  % fix python discrepencies
  mask_min_x = mask_min_x + 1;
  mask_min_y = mask_min_y + 1;
  mask_max_x = mask_min_x + patch_window_width - 1;
  mask_max_y = mask_min_y + patch_window_height - 1;
  
  % bound by image
  mask_min_x = max(mask_min_x, 1)
  mask_min_y = max(mask_min_y, 1)
  mask_max_x = min(mask_max_x, raw_image_width)
  mask_max_y = min(mask_max_y, raw_image_height)

  % create mask & patch
  disp('Extracting patch:');
  mask = zeros(size(raw_image));
  mask(mask_min_y:mask_max_y, mask_min_x:mask_max_x) = 1;
  cropped_array = raw_image(mask_min_y:mask_max_y, mask_min_x:mask_max_x, :);
  disp('Extracting patch done.');

  % calculate collage
  disp('Calculating collage:');
  start_seconds = time();
  haralick_number = 13
  [collage_map, volfeats, Gx, Gy, dominant_orientation_roi, volN]=compute_CoLlAGe2D(raw_image, mask, svd_radius, haralick_number);
  end_seconds = time();
  total_seconds = total_seconds + end_seconds - start_seconds;
  disp('Calculating collage done.');

  % save to random subfolder
  folder_name = ['./results/' num2str(randi(1e5))]
  mkdir(folder_name);
  
  % save parameters
  filename = [folder_name "/parameters.cfg"];
  fid = fopen(filename, "w");
  fprintf(fid, '[DEFAULT]\n');
  fprintf(fid, 'svd_radius = %d\n'         , svd_radius);
  fprintf(fid, 'patch_window_width = %d\n' , patch_window_width);
  fprintf(fid, 'patch_window_height = %d\n', patch_window_height);
  fprintf(fid, 'mask_min_x = %d\n'         , mask_min_x);
  fprintf(fid, 'mask_min_y = %d\n'         , mask_min_y);
  fclose (fid);
  
  % save collage features images
  for f = 1:13
    filename = [folder_name "/collage" num2str(f) "_octave.png"];
    collage_slice = volfeats(:,:,f);
    collage_slice = uint8(255.0 / max(collage_slice(:)) * (collage_slice - min(collage_slice(:))));
    imwrite(collage_slice, filename);
  endfor
  
endfor

% calculate time
seconds_per_collage = total_seconds / N
