function [combineAcc rgbAcc depthAcc] = runCRNN()
% init params
params = initParams();
disp(params);

%% Run RGB
disp('Forward propagating RGB data');
parmas.depth = true;

% load and forward propagate RGB data
if ~exist('results/rgbTrain_test.mat','file')||(~params.isPretrained)
[rgbTrain rgbTest] = forwardProp(params);
save('results/rgbTrain_test.mat','rgbTrain','rgbTest','-v7.3');
else
load('results/rgbTrain_test.mat')
end


% train softmax classifier
disp('Training softmax...');
if ~exist('results/rgbAcc.mat','file')||(~params.isPretrained)
rgbAcc = trainSoftmax(rgbTrain, rgbTest, params);
save('results/rgbAcc.mat','rgbAcc','-v7.3');
else
load('results/rgbAcc.mat')
end
%% Run Depth
disp('Forward propagating depth data');
params.depth = true;

% load and forward propagate depth data
if ~exist('results/depthTrain_test.mat','file')||(~params.isPretrained)
[depthTrain depthTest] = forwardProp(params);
save('results/depthTrain_test.mat','depthTrain','depthTest','-v7.3');
else
load('results/depthTrain_test.mat')
end
% train softmax classifier
if ~exist('results/depthAcc.mat','file')||(~params.isPretrained)
depthAcc = trainSoftmax(depthTrain, depthTest,params);
save('results/depthAcc.mat','depthAcc','-v7.3');
else
load('results/depthAcc.mat')
end
%% Combine RGB + Depth
if ~exist('results/cTrain_test.mat','file')||(~params.isPretrained)
[cTrain cTest] = combineData(rgbTrain, rgbTest, depthTrain, depthTest);
save('results/cTrain_test.mat','cTrain','cTest','-v7.3');
else
load('results/cTrain_test.mat')
end
clear rgbTrain rgbTest depthTrain depthTest;

% test without extra features when combined
if ~exist('results/cAcc.mat','file')||(~params.isPretrained)
params.extraFeatures = false;
combineAcc = trainSoftmax(cTrain, cTest, params);
save('results/cAcc.mat','combineAcc','-v7.3');
else
load('results/cAcc.mat')
end
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
