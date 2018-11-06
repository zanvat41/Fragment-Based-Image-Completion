function [ level_set ] = getLevelSet( confidence_map )
% Compute level set with given confidence map

    confidence_map_std  = std(confidence_map(:));
    confidence_map_mean = mean(confidence_map(:));
    
    level_set = zeros(size(confidence_map));
    for i = 1:size(level_set,1)
        for j = 1:size(level_set,2)
            if confidence_map(i,j) <= confidence_map_mean
                level_set(i,j) = confidence_map(i,j) + confidence_map_std*rand();
            end
        end
    end
end