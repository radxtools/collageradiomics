function [dominant_orientation_roi]=find_orientation_CoLlAGe_2D(Fx_o,Fy_o,k,r,c)

I_gradient_inner=zeros(size(Fx_o, 1), size(Fx_o, 1));



for i= k+1:k+r
    for j= k+1:k+c
        pixel_start_r=i;
        pixel_start_c=j;
        G_x = Fx_o(pixel_start_r-k:pixel_start_r+k,pixel_start_c-k:pixel_start_c+k);
        G_y = Fy_o(pixel_start_r-k:pixel_start_r+k,pixel_start_c-k:pixel_start_c+k);
        G_x = G_x(:);
        G_y = G_y(:);
##        G_x=[];G_y=[];
##        for a= pixel_start_r-k:pixel_start_r+k
##            for b= pixel_start_c-k:pixel_start_c+k
##                
##                G_x=[G_x;Fx_o(a,b)];
##                G_y=[G_y;Fy_o(a,b)];
##                
##            end
##        end
        
       G=[G_x G_y];
       %[U, S, V2]=svd(G);
       V = ComputeV(G);
       dominant_orientation=atan2(V(1,1),V(1,2)); %Find dominant direction
       
       I_gradient_inner(i,j)=dominant_orientation;
    end
end

dominant_orientation_roi= I_gradient_inner(k+1:k+r,k+1:k+c);






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

