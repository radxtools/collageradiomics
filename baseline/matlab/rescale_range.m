function [Iout,N2high] = rescale_range(I,N1,N2,rangedata)
% RESCALE_RANGE Rescale data into a specified range.
%  RESCALE_RANGE(I,N1,N2) rescales the array I so that all elements fall in 
%  the range [N1,N2]. The output is double or single precision.
%
%   See also RESCALE.
%
%JC

% Convert input to double precision
if ~isa(I,'float'),
    I=double(I);
end
Iclass=class(I);
N1=cast(N1,Iclass); N2=cast(N2,Iclass);

% Make sure the data can be rescaled with the current machine precision
if nargin>3,
    rangedata=cast(rangedata,Iclass);
    datarange=max(rangedata(:))-min(rangedata(:));
else
    datarange=max(I(:))-min(I(:));
end
if datarange > eps,
    wantedrange=N2-N1;
    
    Iout = N1+(I-min(I(:)))/(datarange/wantedrange);
else
    Iout=I;
end
if nargout>1, N2high=max(Iout(:)); end
