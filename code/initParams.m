function params = initParams()
% use a small portion of dataset for debugging
params.debug = 1;

% Set the data folder here
params.dataFolder = '../data/';

% set the data split to train and test on 
params.split = 2;

% set the number of first layer CNN filters
params.numFilters = 128;

% set the number of RNN to use
params.numRNN = 64;

% use depth or rgb information
params.depth = false;

% use extra features from segmentation mask
params.extraFeatures = true;

