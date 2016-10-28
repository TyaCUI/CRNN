function [combineAcc rgbAcc depthAcc] = runCRNN()
% init params
params = initParams();
disp(params);

%% Run RGB
disp('Forward propagating RGB data');
parmas.depth = false;

% load and forward propagate RGB data
[rgbTrain rgbTest] = forwardProp(params);

% train softmax classifier
disp('Training softmax...');
rgbAcc = trainSoftmax(rgbTrain, rgbTest, params);

%% Run Depth
disp('Forward propagating depth data');
params.depth = true;

% load and forward propagate depth data
[depthTrain depthTest] = forwardProp(params);

% train softmax classifier
depthAcc = trainSoftmax(depthTrain, depthTest,params);

%% Combine RGB + Depth
[cTrain cTest] = combineData(rgbTrain, rgbTest, depthTrain, depthTest);
clear rgbTrain rgbTest depthTrain depthTest;

% test without extra features when combined
params.extraFeatures = false;
combineAcc = trainSoftmax(cTrain, cTest, params);
return;

function [train test] = forwardProp(params)
% pretrain filters
disp('Pretraining CNN Filters...');
[filters params] = pretrain(params);

% forward prop CNN
disp('Forward prop through CNN...');
[train test] = forwardCNN(filters,params);

% forward prop RNNs
disp('Forward prop through RNN...');
[train test] = forwardRNN(train, test, params);
return;

function [cTrain cTest] = combineData(rgbTrain, rgbTest, depthTrain, depthTest)
% ensure they come from the same file
testCompatability(rgbTrain, depthTrain);
testCompatability(rgbTest, depthTest);

% combine data
cTrain.data = [rgbTrain.data; depthTrain.data];
cTest.data = [rgbTest.data; depthTest.data];

% normalize depth and rgb features independently
m = mean(cTrain.data,2);
s = std(cTrain.data,[],2);
cTrain.data = bsxfun(@rdivide, bsxfun(@minus, cTrain.data,m),s);
cTest.data = bsxfun(@rdivide, bsxfun(@minus, cTest.data,m),s);

% add the labels
cTrain.labels = rgbTrain.labels;
cTest.labels = rgbTest.labels;
return;

function testCompatability(rgb, depth)
assert(length(rgb.file) == length(depth.file));
for i = 1:length(rgb.file)
    assert(strcmp(rgb.file{i}, depth.file{i}));
end

assert(isequal(rgb.labels, depth.labels));
assert(isequal(rgb.labels, depth.labels));
return
