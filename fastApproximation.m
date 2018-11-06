function [ out_im ] = fastApproximation( im, inverse_matte )
% Fast approximation by up/down-scaling

    matte = 1 - inverse_matte;
    
    % set number of levels in pyramid
    L = 3;
    
    % kernel for up/down-scaling
    % values were specified in MATLAB's impyramid documentation
    a = 0.375;
    w = [0.25-a/2,0.25,a,0.25,0.25-a/2];

    Y = ones(size(im));
    for l = L:-1:1
        continue_flag = true;
        while continue_flag
            Y_prev = Y;
            Y = Y.*repmat(matte,1,1,3) + im.*repmat(inverse_matte,1,1,3);
            if l > 1
                for i = 2:l
                    Y = impyramid(Y,'reduce');
                end
                for i = 2:l
                    Y = impyramid(Y,'expand');
                end
            else
                Y = convn(convn(Y, w, 'same'),w','same');
            end

            % resize Y in case of pixels missing during the process of
            % repeatly up/down-scaling
            if size(Y,1) ~= size(Y_prev,1) || size(Y,2) ~= size(Y_prev,2)
                Y = imresize(Y, [size(Y_prev,1),size(Y_prev,2)]);
            end
            
            % if Y has pretty much converged to Y_prev, stop iterating
            if sum(sum(sum((Y - Y_prev).^2))) < 1E-8
                continue_flag = false;
            end
        end
    end
    
    % keep the known area unchanged
    Y = Y.*repmat(matte,1,1,3) + im.*repmat(inverse_matte,1,1,3);
    
    out_im = Y;
end
