% download the toolbox here to use desc_MRELBP
% https://www.mathworks.com/matlabcentral/fileexchange/68427-toolboxdesc?s_tid=prof_contriblnk

%% load data
clearvars
load('soybean_data_10-87.mat');

%%
numTrain = size(x_train, 1);
numVal = size(x_val, 1);
mrelbpTrain = zeros(numTrain, 2592);
mrelbpVal = zeros(numVal, 2592);
descImgVal = cell(numVal, 1);
options.mode = 'nh';

%%
fprintf('Extracting Training Features \n');
for ii = 1:numTrain
    fprintf('Extracting Train Image %d of %d\n', ii, numTrain);
    inImg = bgr2rgb(squeeze(x_train(ii, :, :, :)));
    inImg = rgb2gray(inImg);
    mrelbpTrain(ii,:) = desc_MRELBP(inImg, options);
end

fprintf('Extracting Val Features \n');
for ii = 1:numVal
    fprintf('Extracting Val Image %d of %d\n', ii, numVal);
    inImg = bgr2rgb(squeeze(x_val(ii, :, :, :)));
    inImg = rgb2gray(inImg);
    [mrelbpVal(ii,:), descImgVal{ii}] = desc_MRELBP(inImg, options);
end

save('mreLBP_feats1248Samples8xr_10-87.mat', 'mrelbpTrain', 'mrelbpVal');

%%
function newImage = bgr2rgb(img)
    newImage(:,:,1) = img(:,:,3);
    newImage(:,:,2) = img(:,:,2);
    newImage(:,:,3) = img(:,:,1);
end