function [start_point,end_point]=collage_get_range(mask)
%COLALGE_GET_RANGE: function that gets the start and end indicies of the tumor mask
  start_point =1;
  end_point=size(mask,3);
  final_start = false;
  final_end = false;
  for i=0:size(mask,3) -1
    bot_slice = mask(:,:,start_point +i); 
    top_slice = mask(:,:,end_point -i); 
    if any(bot_slice(:) ~=0 ) && ~final_start
      start_point = start_point +i;
      final_start = true;  
    end
    if any(top_slice(:) ~=0) && ~final_end
      end_point = end_point-i; 
      final_end = true; 
    end
    if final_end && final_start
      break
    end 
  end
  if start_point == 1
    start_point = 2; 
  end
 
  if end_point == size(mask,3)
      end_point =size(mask,3)-1; 
  end

end
