function [feat1_mask,feat2_mask]=compute_CoLlAGe3D(origImage,upperSlice,lowerSlice, tumorMask, winRadius)
%%compute_CoLlAGe3ddnew extract collage intensities and  statistics
%{ 
Args:
    origImage: MxNx1 slice of patient 
    upperSlice: MxNx1 slice of patient should be 1 slice above origImage
    lowerSlice: MxNx1 slice of patient should be 1 slice bellow origImage
    TumorMask: MxNx1  binary mask used to extract features. should overlap
    with origImage tumor region
    winRadius:  window sized to be used: 1 for 3x3 2 for 5x5
Output: 
    feat1_mask= MxNx13 features based on primary orientation 
    feat2_mask= MxNx13 features based on second orientation 
    3rd dimension is for each of the haralick features
%}
%close all
%k=2;

I = flipdim(origImage ,2);
I_upper=flipdim(upperSlice ,2);
I_lower=flipdim(lowerSlice ,2);
mask = flipdim(tumorMask ,2);

[r, c]=size(mask);
mask_new=zeros(r,c);
for i =1:r
    for j = 1:c
        if mask(i,j)>0
            mask_new(i,j)=1;
        end
    end
end
mask=mask_new;
%imshow(mask)
[r_mask,c_mask]=find(mask==1);
c_min=min(c_mask);c_max=max(c_mask);r_min=min(r_mask);r_max=max(r_mask);

x1=r_min;
x2=r_max;
y1=c_min;
y2=c_max;



if (x1-winRadius<1) || (y1-winRadius<1) || (x2+winRadius>size(I,1)) || (y2+winRadius>size(I,2))
    warning('COLLAGE: Gradient would go outside image border. Cannot compute CoLlAGe - Please erode mask. Returning NANs');
    feat1_mask = nan([size(mask),13]);
    feat2_mask = nan([size(mask),13]);
    volfeats = [];
    return
end


%% for middle slice
I2_outer = I(x1-winRadius:x2+winRadius,y1-winRadius:y2+winRadius);
I2_double_outer=im2double(I2_outer);
I2_inner=I(x1:x2,y1:y2);
[r, c]=size(I2_inner);
[r1, c1]=size(I2_double_outer);

%% for upper slice

I2_outer_upper = I_upper(x1-winRadius:x2+winRadius,y1-winRadius:y2+winRadius);
I2_double_outer_upper=im2double(I2_outer_upper);
I2_inner_upper=I_upper(x1:x2,y1:y2);

%% for lower slice
I2_outer_lower = I_lower(x1-winRadius:x2+winRadius,y1-winRadius:y2+winRadius);
I2_double_outer_lower=im2double(I2_outer_lower);
I2_inner_lower=I_lower(x1:x2,y1:y2);
%%

vol_double_outer=zeros(r1,c1,3);
vol_double_outer(:,:,1)=I2_double_outer_lower;
vol_double_outer(:,:,2)=I2_double_outer;
vol_double_outer(:,:,3)=I2_double_outer_upper;
[Lx, Ly, Lz]=gradient(vol_double_outer);

[I_gradient_inner1, I_gradient_inner_mag, dominant_orientation1_roi, dominant_orientation2_roi]=find_orientation_CoLlAGe_3D(Lx,Ly,Lz,winRadius,r,c);
BW1=mask(x1:x2,y1:y2);
feat1_mask= nan([size(mask,1),size(mask,2),13]); 
feat2_mask = nan([size(mask,1),size(mask,2),13]); 

clear volfeats;
haralickfun=@haralick2mex; %haralick2mex or haralick2mexmt 
vol=double(dominant_orientation1_roi);
nharalicks=13;  % Number of Features
bg=-1;   % Background
ws=2*winRadius+1;    % Window Size
hardist=1;   % Distance in a window
harN=64;     % Maximum number of quantization level
volN=round(rescale_range(vol,0,harN-1));   % Quantizing an image
addedfeats=0;  % Feature counter index
volfeats(:,:,addedfeats+(1:nharalicks))=haralickfun(volN,harN,ws,hardist,bg);

for i =1:13 
  feat1_mask(x1:x2,y1:y2,i) = volfeats(:,:,i);
  feat1_mask(:,:,i) = flipdim(feat1_mask(:,:,i),2);
end

clear volfeats;
haralickfun=@haralick2mex;
vol=double(dominant_orientation2_roi);
nharalicks=13;  % Number of Features
bg=-1;   % Background
ws=5;    % Co-occurence Window Fixed at 5
hardist=1;   % Distance in a window
harN=64;     % Maximum number of quantization level
volN=round(rescale_range(vol,0,harN-1));   % Quantizing an image
addedfeats=0;  % Feature counter index
volfeats(:,:,addedfeats+(1:nharalicks))=haralickfun(volN,harN,ws,hardist,bg);
for i =1:13 
  feat2_mask(x1:x2,y1:y2,i) = volfeats(:,:,i);
  feat2_mask(:,:,i) = flipdim(feat2_mask(:,:,i),2);
end
