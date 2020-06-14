folder_name = './results/';
directory = dir(folder_name);

all_percent_differences = [];

for i = 3:length(directory)

  subfolder_name = [folder_name directory(i).name]
  flattened_octave_features = [];
  flattened_python_features = [];
  
  % loop through features
  for f = 1:13

    % load octave results
    filename = [subfolder_name '/octave' num2str(f) '.png'];
    octave_slice = imread(filename);
    flattened_octave_features(f, :) = double(octave_slice(:));
    
    % load python results
    filename = [subfolder_name '/python' num2str(f) '.png'];
    try
      python_slice = imread(filename);
    catch
      % default to adding random noise for testing purposes
      python_slice = imnoise(octave_slice, "poisson");
    end_try_catch
    flattened_python_features(f, :) = double(python_slice(:));
    
  endfor
  
  % calculate percent difference
  flattened_percent_differences = (flattened_python_features - flattened_octave_features) / 256;
  
  % append
  all_percent_differences = [all_percent_differences, flattened_percent_differences];
  
endfor

% display errors as histogram
close all;
figure;
bins = 50;
for f = 1:13
  subplot(3, 5, f);
  hist(all_percent_differences(f, :), bins);
  title(['Collage' num2str(f) ' Mismatch']);
  xlabel('Pixel-wise percent difference');
  ylabel('# of pixels');
endfor
