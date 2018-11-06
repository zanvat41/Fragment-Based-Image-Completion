function [ confidence_map ] = getConfidenceMap( inverse_matte )
% Compute confidence map
    gauss = fspecial('gaussian', 31, 10);

    inverse_matte = padarray(inverse_matte,[15 15],0,'both');
    confidence_map = ones(size(inverse_matte));
    for i = 15+1:size(confidence_map,1)-15
        for j = 15+1:size(confidence_map,2)-15
            if inverse_matte(i,j) < 0.99
                confidence_map(i,j) = sum(sum(inverse_matte(i-15:i+15,j-15:j+15) .* gauss));
            end
        end
    end

    confidence_map = confidence_map(1+15:end-15,1+15:end-15);
end
