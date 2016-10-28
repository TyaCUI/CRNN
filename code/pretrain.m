function [filters params] = pretrain(params)
params.patchOverlap = 0.01;
showFilters = false;

% patches = numPatches x patchSize
patches = getPatches(params); 


%% get whitening info from patches
% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

C = cov(patches);
M = mean(patches);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';

% Now whiten patches before pretraining
patches = bsxfun(@minus, patches, M) * P;
filters = run_kmeans(patches,params.numFilters,100);

if showFilters
    rfSize = 9;
    show_centroids(filters, rfSize);
end

params.whiten.P = P;
params.whiten.M = M;
return;

function patches  = getPatches(params)
% returns patches = numPatches x patchSize
% params should have fields depth, numFilters, patchOverlap

if params.depth
    searchStr = '/*_depthcrop.png';
    channels = 1;
else
    searchStr = '/*_crop.png';
    channels = 3;
end


if params.debug
    numPatches = 1000;
    printStride = 100;
else
    numPatches = 400000;
    printStride = 10000;
end
numWant = numPatches;
rfSize = 9;
patches = zeros(rfSize*rfSize*channels,numPatches);

% grab all categories in data folder
data = [params.dataFolder 'rgbd-dataset'];
categories = dir(data);
categories = categories(randperm(length(categories)));

numHave = 0;
count = 1;
printCount = 0;
while numHave < numWant
    % load this instance
    if isValid(categories(mod(count-1,length(categories))+1).name)
        fileCatName = [data '/' categories(mod(count-1,length(categories))+1).name];
        instance = dir(fileCatName);
        
        
        for instInd = 1:length(instance)
            if isValid(instance(instInd).name) && rand < 0.1
                % subsample instance
                fileInstName = [fileCatName '/' instance(instInd).name ];
                instanceData = getValidInds(dir([fileInstName searchStr]), fileInstName);
                
                instDataInds = randperm(length(instanceData));
                instDataInds = instDataInds(1:round(end*0.01));
                % extract out patches from each image
                for instDataInd = instDataInds
                    fileImgName = [fileInstName '/' instanceData(instDataInd).name];
                    img = imread(fileImgName);
                    
                    startInd = max(strfind(instanceData(instDataInd).name,'_'));
                    maskImgName = [fileInstName '/' instanceData(instDataInd).name(1:startInd) 'maskcrop.png'];
                    mask = imread(maskImgName);
                    
                    % hard code the image resize
                    inSize = 148;
                    sz = [inSize inSize];
                    img = imresize(img,sz);
                    mask = imresize(mask,sz);
                    if params.depth
                        img = depthImageInterpolation(img, mask);
                    end
                    
                    % get all possible patches
                    imgPatches = [];
                    for ch = 1:size(img,3)
                        imgPatches = [imgPatches; im2col(img(:,:,ch), [rfSize rfSize])];
                    end
                    
                    % require an overlab of patch and object
                    segPatches = im2col(mask, [rfSize rfSize]);
                    patchesToKeep = find(sum(segPatches) >= rfSize^2*params.patchOverlap);
                    
                    % subsample all possible patches
                    keepInds = randperm(length(patchesToKeep));
                    numToTake = min(round(length(keepInds)*0.005), numWant-numHave);
                    keepInds = keepInds(1:numToTake);
                    
                    % add patches
                    patches(:,numHave+1:numHave+length(keepInds)) = imgPatches(:,keepInds);
                    numHave = numHave + length(keepInds);
                    if numHave > (printStride*printCount)
                        disp(['Now have ' num2str(numHave) '/' num2str(numWant) ' patches']);
                        printCount = printCount + 1;
                    end
                end
            end
        end
    end
    % go to next file
    count = count + 1;
end
patches = patches';
return;


function fileBool = isValid(name)
fileBool = (~strcmp(name,'.') && ~strcmp(name,'..') && ~strcmp(name,'.DS_Store'));
return;
