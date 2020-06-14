function [ I_gradient_inner1, I_gradient_inner_mag, dominant_orientation1_roi, dominant_orientation2_roi ]=find_orientation_CoLlAGe_3D(Fx_o,Fy_o,Fz_o,k,r,c)

I_gradient_inner1=atan2(Fy_o,Fx_o);
I_gradient_inner2=atan2(Fy_o,Fx_o);
S_inner=size(I_gradient_inner1);
I_gradient_inner_mag=zeros(S_inner(1),S_inner(2));




for i= k+1:k+r
    for j= k+1:k+c
        pixel_start_r=i;
        pixel_start_c=j;
        G_x=[];G_y=[];G_z=[];
        for a= pixel_start_r-k:pixel_start_r+k
            for b= pixel_start_c-k:pixel_start_c+k
                
                G_x=[G_x;Fx_o(a,b)];
                G_y=[G_y;Fy_o(a,b)];
                G_z=[G_z;Fz_o(a,b)];
                
            end
        end
       G=[G_x G_y G_z];
       [U S V]=svd(G);%PCA
       
       % dominant orientation 1
       dominant_orientation1=atan2(V(1,1),V(1,2)); %Find dominant direction
       dominant_magnitude=sqrt(V(1,1)^2+V(1,2)^2); % not required
       I_gradient_inner1(i,j)=dominant_orientation1;
       I_gradient_inner_mag(i,j)=dominant_magnitude; % not required
       
       %dominant orientation2
       dominant_orientation2=atan2(V(1,3),sqrt(V(1,1)^2+V(1,2)^2)); %Find dominant direction
       I_gradient_inner2(i,j)=dominant_orientation2;
       
       
    end
end

dominant_orientation1_roi= I_gradient_inner1(k+1:k+r,k+1:k+c);
dominant_orientation2_roi= I_gradient_inner2(k+1:k+r,k+1:k+c);






% 
% for i= k+1:29+k+1
%     for j= k+1:29+k+1
%        pixel_start_r=i;
%        pixel_start_c=j;
%        G_x=[];G_y=[];
%        for a= pixel_start_r-k:pixel_start_r+k
%            for b= pixel_start_c-k:pixel_start_c+k
%                
%                G_x=[G_x;Fx_o(a,b)];
%                G_y=[G_y;Fy_o(a,b)];
%                
%            end
%        end
%        G=[G_x G_y];
%        [U S V]=svd(G);%PCA
%        dominant_orientation=atan2(V(1,1),V(1,2)); %Find dominant direction
%        dominant_magnitude=sqrt(V(1,1)^2+V(1,2)^2); % not required
%        I_gradient_inner(i,j)=dominant_orientation;
%        I_gradient_inner_mag(i,j)=dominant_magnitude; % not required
%     end
% end


end

