totalTic = tic;

%% load image; load/create inverse matte
% Modified the input data for different test

%im = im2double(imread('../Input/lake.jpg'));
%imname = 'lake';
%im = im2double(imread('../Input/valley.jpg'));
%imname = 'valley';
%im = im2double(imread('../Input/roadmark.jpg'));
%imname = 'roadmark';
%im = im2double(imread('../Input/lake2.jpg'));
%imname = 'lake2';
%inverse_matte = ones(size(im,1),size(im,2));
%inverse_matte(100:129,112:141) = 0;
%inverse_matte(59:88,172:201) = 0;
%inverse_matte(211:240,161:190) = 0;

%im = im2double(imread('../Input/text.png'));
%imname = 'text';
%inverse_matte = ones(size(im,1),size(im,2));
%inverse_matte(rgb2gray(im)==1) = 0;

im = im2double(imread('../Input/starrynight.jpg'));
imname = 'sn';
inverse_matte = floor(im2double(rgb2gray(imread('../Input/starrynight_inverse_matte.jpg')))+0.5);

x1 = find(sum(1-inverse_matte,2)~=0,1);
y1 = find(sum(1-inverse_matte,1)~=0,1);
x2 = find(sum(1-inverse_matte,2)~=0,1,'last');
y2 = find(sum(1-inverse_matte,1)~=0,1,'last');

%% fast approximation
imwrite(im.*repmat(inverse_matte,1,1,3), strcat('../Output/',imname,'_init.jpg'));
figure, imshow(im.*repmat(inverse_matte,1,1,3));
approx_im = fastApproximation(im, inverse_matte);
figure, imshow(approx_im);

%% compute confidence map
confidence_map = getConfidenceMap(inverse_matte);
%figure, imshow(confidence_map);

%% compute level set
level_set = getLevelSet(confidence_map);
%figure, imshow(level_set);

%% initialize nnf
NNF_set = cell(1,8);
for i = 1:8
    NNF_set{i} = 0;
end

%% do the iterations
iter = 0;
while mean(confidence_map(:)) < 0.9999 || min(inverse_matte(:)) < 0.5
    iter = iter + 1;
    
    %% search for matching patch
    % features for distance calculation
    features = zeros(size(approx_im,1),size(approx_im,2),8);
    features(:,:,1:3) = approx_im;
    features(:,:,4) = rgb2gray(approx_im);
    features(:,:,5) = conv2(features(:,:,4),[-0.5,0,0.5],'same').*inverse_matte;
    features(:,:,6) = conv2(features(:,:,4),[-0.5,0,0.5]','same').*inverse_matte;
    features(:,:,7) = conv2(features(:,:,4),[-0.5,0,0;0,0,0;0,0,0.5],'same').*inverse_matte;
    features(:,:,8) = conv2(features(:,:,4),[0,0,-0.5;0,0,0;0.5,0,0]','same').*inverse_matte;

    % retrieve the image coordinates of the pixel 
    % with maximal value in level_set
    [M, max_i] = max(level_set);
    [M, max_j] = max(M);
    max_i = max_i(max_j);
    
    % calculate patch size  
    rad = zeros(3, 1);
    rad(1) = 4;
    rad(2) = 7;
    rad(3) = 10;
    patch_params = zeros(size(rad,1),1);

    averageGradient=Inf;
    isBoundary = true;
    for i=1:size(rad,1)
        % check boundary pixels
        if (max_i-rad(i)>=1 && max_i+rad(i)<=size(features,1) && max_j-rad(i)>=1 && max_j+rad(i)<=size(features,2))
            currentPatch = abs(features(max_i-rad(i):max_i+rad(i), max_j-rad(i):max_j+rad(i),5:8));
            currInverseMatte = inverse_matte(max_i-rad(i):max_i+rad(i), max_j-rad(i):max_j+rad(i));
        else
            continue;
        end
        confidentCurrPatch=currentPatch(repmat(currInverseMatte>0.95,1,1,4));
        if mean(confidentCurrPatch(:)) < (1+(0.01*(4.5+(1/sqrt(averageGradient)))))*averageGradient 
            averageGradient = mean(confidentCurrPatch);
            maxPatchIndex = i;
            isBoundary = false;
        end
    end
    if isBoundary
        [~,maxPatchIndex]=min(patch_params);
    end
    patch_size = rad(maxPatchIndex).*2+1;
    
    % increase border
    % [x1,y1;x2,y2] comes from the missing region we selected
    unknown_bbox = [x1-patch_size,y1-patch_size;x2+patch_size,y2+patch_size]; 
    if unknown_bbox(1,1) < 1, unknown_bbox(1,1) = 1; end
    if unknown_bbox(1,2) < 1, unknown_bbox(1,2) = 1; end
    if unknown_bbox(2,1) > size(features,1), unknown_bbox(2,1) = size(features,1); end
    if unknown_bbox(2,2) > size(features,2), unknown_bbox(2,2) = size(features,2); end
    
    % get a matching patch
    [target_patch, source_patch, matte_sp, sp_i, sp_j, NNF_set] = ...
            getMatchingPatch(features,unknown_bbox,max_i,max_j,inverse_matte,patch_size,NNF_set);
    

    %% composite patches
    hps = floor(patch_size/2); % half of patch size
    matte_tp = inverse_matte(max_i-hps:max_i+hps,max_j-hps:max_j+hps);
    [composite_patch, matte_patch] = compositePatch(target_patch(:,:,1:3), source_patch, matte_tp, matte_sp, 3);


    %% insert composite patch
    approx_im = approx_im .* repmat(inverse_matte,1,1,3);
    approx_im(max_i-hps:max_i+hps,max_j-hps:max_j+hps,:) = composite_patch;
    inverse_matte(max_i-hps:max_i+hps,max_j-hps:max_j+hps) = matte_patch;
    
    %% fast approximation
    imwrite(approx_im.*repmat(inverse_matte,1,1,3), strcat('../Output/',imname,'_progress_',int2str(iter),'.jpg'));
    figure, imshow(approx_im.*repmat(inverse_matte,1,1,3));
    approx_im = fastApproximation(approx_im, inverse_matte);

    %% recompute confidence map
    confidence_map = getConfidenceMap(inverse_matte);

    %% recompute level set
    level_set = getLevelSet(confidence_map);
    
    
end

imwrite(approx_im.*repmat(inverse_matte,1,1,3), strcat('../Output/',imname,'_finished.jpg'));
figure, imshow(approx_im);

toc(totalTic);
