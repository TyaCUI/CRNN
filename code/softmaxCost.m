function [cost,grad] = softmaxCost(X,x,y,lambda,decodeInfo,params)
% x is (numFeatures x numTrain)
theta = stack2param(X,decodeInfo);
 
pred = exp(theta*x); % k by n matrix with all calcs needed
pred = bsxfun(@rdivide,pred,sum(pred));
m = length(y);
truth_inds = sub2ind(size(pred),y,1:m);
cost = sum(log(pred(truth_inds)))/m;
cost = -cost + (lambda/2)*sum(sum(theta.^2));

truth = zeros(size(pred));
truth(truth_inds) = 1;
error = pred - truth;
grad = (error*x')/m;
grad = grad + lambda*theta;


grad = param2stack(grad);
end