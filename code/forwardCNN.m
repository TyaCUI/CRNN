function [train test] = forwardCNN(filters, params)
%% init the data containers
nf = params.numFilters;
% hard code final size
fi = 27;
fj = 27;

numExtra = 3;
train = struct('data',zeros(100,nf,fi,fj),'labels',[],'count',0,'extra',zeros(100,numExtra),'file',[]);
test = struct('data',zeros(100,nf,fi,fj),'labels',[],'count',0,'extra',zeros(100,numExtra),'file',[]);

%% load the split information
load([params.dataFolder 'splits.mat'],'splits');
% testInstances specifies which instance from each class will used for testing
testInstances = splits(:,params.split);

% grab all categories in data folder
data = [params.dataFolder '/rgbd-dataset'];
categories = dir(data);
if params.debug
    numCategories = 5;
else
    numCategories = length(categories);
end
catNum = 0;
for catInd = 1:numCategories
    if mod(catInd,10) == 0
        disp(['---Category: ' num2str(catInd) ' out of ' num2str(numCategories) '---']);
    end
    if isValid(categories(catInd).name)
        catNum = catNum+1;
        % grab all instances within this category
        fileCatName = [data '/' categories(catInd).name];
        instance = dir(fileCatName);
        for instInd = 1:length(instance)
            if isValid(instance(instInd).name)
                % check if a testing instance then take data
                fileInstName = [fileCatName '/' instance(instInd).name ];
                if testInstances(catNum) == str2double(instance(instInd).name(regexp(instance(instInd).name,'[0-9]')))
                    test = addInstance(fileInstName, filters, catNum, test, params);
                else
                    train = addInstance(fileInstName, filters, catNum, train, params);
                end
            end
        end
    end
end

train = cutData(train);
test = cutData(test);
return

function fileBool = isValid(name)
fileBool = (~strcmp(name,'.') && ~strcmp(name,'..') && ~strcmp(name,'.DS_Store'));
return;

function data = addInstance(fileInstName, filters, catNum, data, params)
if params.depth
    searchStr = '/*_depthcrop.png';
else
    searchStr = '/*_crop.png';
end

% grab image names from this instance
instanceData = getValidInds(dir([fileInstName searchStr]), fileInstName);

if params.debug
    subSampleInds = 1:5:25;
else
    subSampleInds = 1:5:length(instanceData);
end

% set the labels
data.labels = [data.labels ones(1,length(subSampleInds))*catNum];
for imgInd = subSampleInds
    data.count = data.count + 1;
    
    % read in our file from disk
    fileImgName = [fileInstName '/' instanceData(imgInd).name];
    img = imread(fileImgName);
    startInd = max(strfind(instanceData(imgInd).name,'_'));
    maskImgName = [fileInstName '/' instanceData(imgInd).name(1:startInd) 'maskcrop.png'];
    mask = imread(maskImgName);
    if params.depth
        img = depthImageInterpolation(img, mask);
    end
    
    % hard code the image resize
    inSize = 148;
    img = double(imresize(img,[inSize inSize]));
  	
    % extract random features
    fim = extractFeatures(img, filters, params);
    
    % add these features to data
    if data.count > size(data.data,1)
        data.data(end*2,end,end) = 0;
        data.extra(end*2,end) = 0;
    end
    data.data(data.count,:) = fim(:);
    
    % add to extra featuers
    [rows cols] = ind2sub(size(mask), find(mask>0));
    data.extra(data.count,:) = [(max(rows)-min(rows)) (max(cols)-min(cols)) sum(mask(:))];
    
    % add for sanity check
    data.file{end+1} = instanceData(imgInd).name(1:startInd);
end
return


function data = cutData(data)
assert(length(data.labels) == data.count);
data.data = data.data(1:data.count,:,:,:);
data.extra = data.extra(1:data.count,:);
return




