function [ patch_out, matte_out ] = compositePatch( tp, sp, matte_tp, matte_sp, scales )
    tp_pyramid = getLaplacianPyramid(tp, scales);
    sp_pyramid = getLaplacianPyramid(sp, scales);
    ones_pyramid = getLaplacianPyramid(ones(size(tp,1),size(tp,2)), scales);
    
    matte_tp_pyramid = cell(1,scales);
    matte_tp_pyramid{1} = matte_tp;
    matte_sp_pyramid = cell(1,scales);
    matte_sp_pyramid{1} = matte_sp;
    inverse_matte_tp_pyramid = cell(1,scales);
    inverse_matte_tp_pyramid{1} = 1 - matte_tp;
    
    for scale = 2:scales
        matte_tp_pyramid{scale} = impyramid(matte_tp_pyramid{scale-1},'reduce');
        inverse_matte_tp_pyramid{scale} = impyramid(inverse_matte_tp_pyramid{scale-1},'reduce');
        matte_sp_pyramid{scale} = impyramid(matte_sp_pyramid{scale-1},'reduce');
    end

    p_out = cell(1,scales);
    m_out = cell(1,scales);
    for scale = 1:scales
        p_out{scale} = tp_pyramid{scale}.*repmat(matte_tp_pyramid{scale},1,1,3) + ...
                sp_pyramid{scale}.*repmat(matte_sp_pyramid{scale},1,1,3).*repmat(inverse_matte_tp_pyramid{scale},1,1,3);
        m_out{scale} = ones_pyramid{scale}.*matte_tp_pyramid{scale} + ...
                ones_pyramid{scale}.*matte_sp_pyramid{scale}.*inverse_matte_tp_pyramid{scale};
    end

    patch_out = reconstruct(p_out);
    matte_out = reconstruct(m_out);
end

function [ pyramid ] = getLaplacianPyramid( im, scales )
    pyramid = cell(1,scales);
    pyramid{1} = im;
    for scale = 1:scales-1
        pyramid{scale+1} = impyramid(pyramid{scale},'reduce');
        im_exp = impyramid(pyramid{scale+1},'expand');
        if size(im_exp,1) ~= size(pyramid{scale},1) || size(im_exp,2) ~= size(pyramid{scale},2)
            im_exp = imresize(im_exp, [size(pyramid{scale},1),size(pyramid{scale},2)]);
        end
        pyramid{scale} = pyramid{scale} - im_exp;
    end
    
end

function [ im_out ] = reconstruct( pyramid )
    for scale = length(pyramid)-1:-1:1
        im_exp = impyramid(pyramid{scale+1},'expand');
        if size(im_exp,1) ~= size(pyramid{scale},1) || size(im_exp,2) ~= size(pyramid{scale},2)
            im_exp = imresize(im_exp, [size(pyramid{scale},1),size(pyramid{scale},2)]);
        end
        pyramid{scale} = pyramid{scale} + im_exp;
    end
    im_out = pyramid{1};

end

