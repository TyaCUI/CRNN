function percentCorrect = trainSoftmax(train,test, params)
% train.data must in a [features x numTrain] format
% train.labels must be labeled in  1 to numLabels format
addpath('../minFunc');

if params.debug,
    options.maxIter = 4;
else
    options.maxIter = 350;
end

if params.extraFeatures
    [train test] = addExtraFeatures(train, test);
end

% init theta
k = max(train.labels);
n = size(train.data,1);
Wcat = 0.005 * randn(k,n);

% train
options.Method = 'lbfgs';
options.display = 'on';
lambda = 1e-8;
[X decodeInfo] = param2stack(Wcat);
X = minFunc(@softmaxCost, X, options, train.data, train.labels, lambda,decodeInfo, params);
Wcat = stack2param(X,decodeInfo);

% test
percentCorrect = softmaxTest(Wcat,test.data, test.labels);
disp(['percent correct: ' num2str(percentCorrect)]);
return


function [train test] = addExtraFeatures(train, test)
extraStd = 12;
m = mean(train.extra);
s = std(train.extra)/extraStd;
train.extra = bsxfun(@rdivide,bsxfun(@minus,train.extra,m),s);
test.extra = bsxfun(@rdivide,bsxfun(@minus,test.extra,m),s);

train.data = [train.data;train.extra'];
test.data = [test.data;test.extra'];
return

