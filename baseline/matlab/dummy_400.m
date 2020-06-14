vol=ones(20);
vol(:) = 1:numel(vol);
harN = 64;
volN = round(rescale_range(vol, 0, harN-1))';

haralick_window_size = 20;
haralickfun=@haralick2mex;
har_output = haralickfun(volN, harN, haralick_window_size, 1, -1);

close all; clf;
for i = 1:13
  subplot(3,5,i);
  imagesc(har_output(:,:,i));
  colormap(jet); colorbar; title(i);
end
