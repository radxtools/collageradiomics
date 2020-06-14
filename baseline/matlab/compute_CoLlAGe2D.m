
function [collage_map, volfeats, Gx, Gy, dominant_orientation_roi, volN]=compute_CoLlAGe2D(origImage, tumorMask, winRadius,haralick_number)
%close all
% Med_entropy=[];
% Mean_entropy=[];
% index_array=[];
% val_hist=[];
% feature_concat=[];


% 
% %origImage=case19_2.standardized_volume_T1(:,:,4);
% I = flipdim(origImage ,2);
% 
% %tumorMask=case19_2.mask_T1(:,:,4);
% mask = flipdim(tumorMask ,2);

%change as necessary
I=origImage;
mask=tumorMask;

%imshow(mask)
[r_mask,c_mask]=find(mask==1);
c_min=min(c_mask);c_max=max(c_mask);r_min=min(r_mask);r_max=max(r_mask);

x1=r_min;
x2=r_max;
y1=c_min;
y2=c_max;

% h = imfreehand; % now pick ROI
% BW = createMask(h); % get BW mask for that ROI
% pos = getPosition(h);
% x1 =  round(min(pos(:,2)));
% y1 =  round(min(pos(:,1)));
% x2 =  round(max(pos(:,2)));
% y2 =  round(max(pos(:,1)));
%I2 = I.*uint8(BW); % apply mask to image

disp('Calculating gradient:');

if (x1-winRadius<1) || (y1-winRadius<1) || (x2+winRadius>size(I,1)) || (y2+winRadius>size(I,2))
    warning('COLLAGE: Gradient would go outside image border. Cannot compute CoLlAGe - Please erode mask. Returning NANs');
    collage_map = nan([size(mask),max(haralick_number)]);
    volfeats = [];
    xxx = 1; %pause; testing
    return
end

I2_outer = I(max(x1-winRadius,1):min(x2+winRadius,size(I,1)),max(y1-winRadius,1):min(y2+winRadius,size(I,2)));
I2_double_outer=im2double(I2_outer);
I2_inner=I(x1:x2,y1:y2);
[r, c]=size(I2_inner);
[Gx, Gy]=gradient(I2_double_outer);

%[Fx, Fy]=gradient(I2_double_outer);
%K_roi=sqrt(Fx.*Fx+Fy.*Fy);

disp('Calculating gradient done.');
disp('Calculating orientation:');

[dominant_orientation_roi]=find_orientation_CoLlAGe_2D(Gx,Gy,winRadius,r,c);

disp('Calculating orientation done.');
disp('Calculating Haralick:');

BW1=mask(x1:x2,y1:y2);
BW1=double(BW1);

% figure(1)% plot arrows on intensities
% 
% imagesc(I2_inner); colormap(gray)
% hold on
% for i =1:3:r
%     for j= 1:3:c
%            %arrowlength=4*K_roi_outer(i,j)/max(max(K_roi_outer));
%            if BW1(i,j)==1
%            arrowlength=2;
%            h= quiver(j, i, cos(dominant_orientation_roi(i,j)), sin(-dominant_orientation_roi(i,j)),arrowlength,'b','linewidth',3);
% %            pause
%            adjust_quiver_arrowhead_size(h,10);
%            end
%            hold on
%     end
% 
% end
% hold off


% Find co-occurrence features of orientation
%clear volfeats;
haralickfun=@haralick2mex;
vol=double(dominant_orientation_roi);



   
nharalicks=13;  % Number of Features
bg=-1;   % Background-1
ws=2*winRadius+1;    % Window Size
hardist=1;   % Distance to search in a window
harN=64;     % Maximum number of quantization level 64
volN=round(rescale_range(vol,0,harN-1));   % Quantizing an image
% volN(~volN) = 1; 
% volN
addedfeats=0;  % Feature counter index

volfeats = zeros(size(volN, 1), size(volN, 2), 13);
%% 
volfeats(:,:,addedfeats+(1:nharalicks))=haralickfun(volN,harN,ws,hardist,bg);

%Plot arrows on orientation-CM features
%figure(3)
%imagesc(volfeats(:,:,haralick_number).*BW1);
%pause
%hold on
% for i =1:2:r
%     for j= 1:2:c
%            if BW1(i,j)==1
%            %arrowlength=4*K_roi_outer(i,j)/max(max(K_roi_outer));
%            arrowlength=1.5;
%            h= quiver(j, i, cos(dominant_orientation_roi(i,j)), sin(-dominant_orientation_roi(i,j)),arrowlength,'b','linewidth',3);
% %            pause
%            adjust_quiver_arrowhead_size(h, 5);
%            hold on
%            end
%     end
% 
% end
% 
% 
%colormap(jet)
%axis off
%pause

collage_map = nan([size(mask),max(haralick_number)]);
collage_map(x1:x2,y1:y2,haralick_number) = volfeats(:,:,haralick_number);
% OCM_feature=volfeats(:,:,haralick_number);
% for i=1:r
%     for j=1:c
%          if BW1(i,j)==1
%              feature_concat=[feature_concat OCM_feature(i,j)];
%          end
%     end
% end
% 
% 

disp('Calculating Haralick done.');
