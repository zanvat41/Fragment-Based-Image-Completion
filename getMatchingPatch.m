function [ t_patch, s_patch, sp_inverse_matte, sp_i, sp_j, NNF_set ] = ...
    getMatchingPatch( features, unknown_bbox, pi, pj, inverse_matte, ps, NNF_set )
% Find a matching patch for given target patch using nearest-neighbor field 

hps = floor(ps/2);

tp_features = features(unknown_bbox(1,1):unknown_bbox(2,1),unknown_bbox(1,2):unknown_bbox(2,2),:);
tp_inverse_matte = floor(inverse_matte(pi-hps:pi+hps,pj-hps:pj+hps)+0.05);
inverse_matte_expanded_straight = floor(imgaussfilt(inverse_matte,6));
t_im_inverse_matte_expanded = inverse_matte_expanded_straight(unknown_bbox(1,1):unknown_bbox(2,1),unknown_bbox(1,2):unknown_bbox(2,2));

adj_pi = pi - unknown_bbox(1,1) + 1;
adj_pj = pj - unknown_bbox(1,2) + 1;

t_patch = features(pi-hps:pi+hps,pj-hps:pj+hps,1:3);

feature_set = cell(1,8);
sp_inverse_matte_set = cell(1,8);
for orient = 1:8
    feature_set{orient} = features;
    sp_inverse_matte_set{orient} = inverse_matte;
    switch orient
        case 2
            temp_features = zeros(size(feature_set{orient},2),size(feature_set{orient},1),size(features,3));
            for i = 1:size(features,3)
                temp_features(:,:,i) = imrotate(feature_set{orient}(:,:,i),90);
            end
            feature_set{orient} = temp_features;
            sp_inverse_matte_set{orient} = imrotate(sp_inverse_matte_set{orient},90);
        case 3
            temp_features = zeros(size(feature_set{orient},1),size(feature_set{orient},2),size(features,3));
            for i = 1:size(features,3)
                temp_features(:,:,i) = imrotate(feature_set{orient}(:,:,i),180);
            end
            feature_set{orient} = temp_features;
            sp_inverse_matte_set{orient} = imrotate(sp_inverse_matte_set{orient},180);
        case 4
            temp_features = zeros(size(feature_set{orient},2),size(feature_set{orient},1),size(features,3));
            for i = 1:size(features,3)
                temp_features(:,:,i) = imrotate(feature_set{orient}(:,:,i),270);
            end
            feature_set{orient} = temp_features;
            sp_inverse_matte_set{orient} = imrotate(sp_inverse_matte_set{orient},270);
        case 5
            temp_features = zeros(size(feature_set{orient},1),size(feature_set{orient},2),size(features,3));
            for i = 1:size(features,3)
                temp_features(:,:,i) = flip(feature_set{orient}(:,:,i), 1);
            end
            feature_set{orient} = temp_features;
            sp_inverse_matte_set{orient} = flip(sp_inverse_matte_set{orient}, 1);
        case 6
            temp_features = zeros(size(feature_set{orient},2),size(feature_set{orient},1),size(features,3));
            for i = 1:size(features,3)
                temp_features(:,:,i) = flip(imrotate(feature_set{orient}(:,:,i),90), 1);
            end
            feature_set{orient} = temp_features;
            sp_inverse_matte_set{orient} = flip(imrotate(sp_inverse_matte_set{orient},90), 1);
        case 7
            temp_features = zeros(size(feature_set{orient},1),size(feature_set{orient},2),size(features,3));
            for i = 1:size(features,3)
                temp_features(:,:,i) = flip(imrotate(feature_set{orient}(:,:,i),180), 1);
            end
            feature_set{orient} = temp_features;
            sp_inverse_matte_set{orient} = flip(imrotate(sp_inverse_matte_set{orient},180), 1);
        case 8
            temp_features = zeros(size(feature_set{orient},2),size(feature_set{orient},1),size(features,3));
            for i = 1:size(features,3)
                temp_features(:,:,i) = flip(imrotate(feature_set{orient}(:,:,i),270), 1);
            end
            feature_set{orient} = temp_features;
            sp_inverse_matte_set{orient} = flip(imrotate(sp_inverse_matte_set{orient},270), 1);
    end
end

best_offset = Inf;
for orient = 1:8
    
    inverse_matte_expanded = floor(imgaussfilt(sp_inverse_matte_set{orient},6));
    
    [NNF,offsets] = getNNF(tp_features, feature_set{orient}, ps, inverse_matte_expanded, tp_inverse_matte, t_im_inverse_matte_expanded);
    NNF_set{orient} = NNF;
    
    offset = offsets(adj_pi,adj_pj);
    
    if offset < best_offset
        sp_i = NNF(adj_pi,adj_pj,1);
        sp_j = NNF(adj_pi,adj_pj,2);
        s_patch = feature_set{orient}(sp_i-hps:sp_i+hps,sp_j-hps:sp_j+hps,1:3);
        sp_inverse_matte = sp_inverse_matte_set{orient}(sp_i-hps:sp_i+hps,sp_j-hps:sp_j+hps);
    end
end

end

function [ NNF, offsets ] = getNNF( t_im, s_im, ps, inverse_matte_expanded, tp_inverse_matte, t_im_inverse_matte_expanded)
% Create NNF for target image to map patches to source image

hps = floor(ps/2);

tp_inverse_matte_four = repmat(tp_inverse_matte,1,1,4);

% pad t_im with NaNs
t_im_NaN = padarray(t_im, [hps,hps], NaN, 'both');

t_im_inverse_matte_expanded_NaN = padarray(t_im_inverse_matte_expanded, [hps,hps], NaN, 'both');

% initialize NNF randomly to the size of t_im
% with coordinate values within s_im
NNF = rand(size(t_im,1),size(t_im,2),2);
NNF(:,:,1) = ceil((size(s_im,1)-2*hps) * NNF(:,:,1))+hps;
NNF(:,:,2) = ceil((size(s_im,2)-2*hps) * NNF(:,:,2))+hps;
NNF = padarray(NNF, [hps,hps], NaN, 'both');

% initialize offsets (calculate L1-distances from NNF)
% ignore NaN values
offsets = zeros(size(t_im));
offsets = padarray(offsets, [hps,hps], NaN, 'both');
for i = hps+1:size(t_im_NaN,1)-hps
    for j = hps+1:size(t_im_NaN,2)-hps
        t_patch = t_im_NaN(i-hps:i+hps,j-hps:j+hps,:);
        
        si = NNF(i,j,1);
        sj = NNF(i,j,2);
        
        % re-initialize if (si,sj) is in unknown_bbox
        while isnan(si) || isnan(sj) || ...
              si < 1+hps || si > size(s_im,1)-hps || ...
              sj < 1+hps || sj > size(s_im,2)-hps || ...
              mean(mean(inverse_matte_expanded(si-hps:si+hps,sj-hps:sj+hps))) < 0.99
            NNF(i,j,1) = randi([1+hps,size(s_im,1)-hps]);
            NNF(i,j,2) = randi([1+hps,size(s_im,2)-hps]);
            si = NNF(i,j,1);
            sj = NNF(i,j,2);
        end
        
        
        s_patch = s_im(si-hps:si+hps,sj-hps:sj+hps,:);
        diff = t_patch-s_patch;
        diff(:,:,5:8) = diff(:,:,5:8) .* tp_inverse_matte_four;
        diff = diff(~isnan(diff(:)));
        offsets(i,j) = norm(diff,1);
    end
end


% ITERATION
max_iter = 5;

for iter = 1:max_iter

    if mod(iter,2) == 1
        % odd iteration
        start_i = hps+1;
        end_i   = size(t_im_NaN,1)-hps;
        start_j = hps+1;
        end_j   = size(t_im_NaN,2)-hps;
        step    = 1;
    else
        % even iteration
        start_i = size(t_im_NaN,1)-hps;
        end_i   = hps+1;
        start_j = size(t_im_NaN,2)-hps;
        end_j   = hps+1;
        step    = -1;
    end
    
    
    for i = start_i:step:end_i
        for j = start_j:step:end_j
            if t_im_inverse_matte_expanded_NaN(i,j) == 1
                continue;
            end
            

            % at every t_im(i,j), t_im(i-step,j), t_im(i, j-step)
            % get patch offset
            if mod(iter,2) == 1
                % odd iteration
                blue_off  = offsets(i,j);
                red_off   = offsets(max(1,i-step),j);
                green_off = offsets(i,max(1,j-step));
            else
                % even iteration
                blue_off  = offsets(i,j);
                red_off   = offsets(min(size(s_im,1),i-step),j);
                green_off = offsets(i,min(size(s_im,2),j-step));
            end
            
            % get the minimum
            [~, min_ind] = min([blue_off,red_off,green_off]);
            
            if min_ind == 1
                % no change
            elseif min_ind == 2
                % use (NNF(i-step,j,1)+step,NNF(i-step,j,2)) for blue
                if mod(iter,2) == 1
                    NNF(i,j,:) = [min(size(s_im,1)-hps,NNF(i-step,j,1)+step),NNF(i-step,j,2)];
                else
                    NNF(i,j,:) = [max(1+hps,NNF(i-step,j,1)+step),NNF(i-step,j,2)];
                end
            elseif min_ind == 3
                % use (NNF(i,j-step,1),NNF(i,j-step,2)+step) for blue
                if mod(iter,2) == 1
                    NNF(i,j,:) = [NNF(i,j-step,1),min(size(s_im,2)-hps,NNF(i,j-step,2)+step)];
                else
                    NNF(i,j,:) = [NNF(i,j-step,1),max(1+hps,NNF(i,j-step,2)+step)];
                end
            end
            
            % recalculate offset if NNF is changed above
            if min_ind == 2 || min_ind == 3
                t_patch = t_im_NaN(i-hps:i+hps,j-hps:j+hps,:);
                si = NNF(i,j,1);
                sj = NNF(i,j,2);
                
                % if (si,sj) is in unknown_bbox
                % switch back to red's or green's coordinates
                while isnan(si) || isnan(sj) || ...
                      si < 1+hps || si > size(s_im,1)-hps || ...
                      sj < 1+hps || sj > size(s_im,2)-hps || ...
                      mean(mean(inverse_matte_expanded(si-hps:si+hps,sj-hps:sj+hps))) < 0.99
                    if min_ind == 2
                        NNF(i,j,:) = NNF(max(1,i-step),j,:);
                    elseif min_ind == 3
                        NNF(i,j,:) = NNF(i,max(1,j-step),:);
                    end
                    si = NNF(i,j,1);
                    sj = NNF(i,j,2);
                end
        
                s_patch = s_im(si-hps:si+hps,sj-hps:sj+hps,:);
                diff = t_patch-s_patch;
                diff(:,:,5:8) = diff(:,:,5:8) .* tp_inverse_matte_four;
                diff = diff(~isnan(diff(:)));
                offsets(i,j) = norm(diff,1);
            end


            % Random search
            k = 1;
            w = squeeze([size(s_im,1);size(s_im,2)]);
            alpha = 0.5;
            old_nn = squeeze(NNF(i,j,:));
            old_offset = offsets(i,j);
            best_min_val = old_offset;
            while w(1)*(alpha^k) > 1 && w(2)*(alpha^k) > 1
                new_nn = floor(old_nn + (alpha^k)*(w.*(2*(squeeze(rand(2,1))-0.5))));
                
                % resize new_nn to fit size of s_im
                if new_nn(1) < hps+1, new_nn(1) = hps+1; end
                if new_nn(1) > size(s_im,1)-hps, new_nn(1) = size(s_im,1)-hps; end
                if new_nn(2) < hps+1, new_nn(2) = hps+1; end
                if new_nn(2) > size(s_im,2)-hps, new_nn(2) = size(s_im,2)-hps; end
                
                % calculate offset for this new_nn
                t_patch = t_im_NaN(i-hps:i+hps,j-hps:j+hps,:);
                si = new_nn(1);
                sj = new_nn(2);
                
                % re-initialize if (si,sj) is in unknown_bbox
                while isnan(si) || isnan(sj) || ...
                      si < 1+hps || si > size(s_im,1)-hps || ...
                      sj < 1+hps || sj > size(s_im,2)-hps || ...
                      mean(mean(inverse_matte_expanded(si-hps:si+hps,sj-hps:sj+hps))) < 0.99
                    new_nn = floor(old_nn + (alpha^k)*(w.*(2*(squeeze(rand(2,1))-0.5))));

                    % resize new_nn to fit size of s_im
                    if new_nn(1) < hps+1, new_nn(1) = hps+1; end
                    if new_nn(1) > size(s_im,1)-hps, new_nn(1) = size(s_im,1)-hps; end
                    if new_nn(2) < hps+1, new_nn(2) = hps+1; end
                    if new_nn(2) > size(s_im,2)-hps, new_nn(2) = size(s_im,2)-hps; end

                    % calculate offset for new_nn
                    t_patch = t_im_NaN(i-hps:i+hps,j-hps:j+hps,:);
                    si = new_nn(1);
                    sj = new_nn(2);
                end
                
                s_patch = s_im(si-hps:si+hps,sj-hps:sj+hps,:);
                diff = t_patch-s_patch;
                diff(:,:,5:8) = diff(:,:,5:8) .* tp_inverse_matte_four;
                diff = diff(~isnan(diff(:)));
                new_offset = norm(diff,1);
                
                if new_offset < best_min_val
                    best_min_val = new_offset;
                    NNF(i,j,:) = new_nn;
                    offsets(i,j) = new_offset;
                end
                
                k = k+1;
            end
            
        end
    end

end
    
% delete NAN values padding
NNF = NNF(1+hps:end-hps,1+hps:end-hps,:);
offsets = offsets(1+hps:end-hps,1+hps:end-hps,:);
end
