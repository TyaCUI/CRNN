function [percentCorrect, y_predict, confidence] = softmaxTest(theta,x,y)

correct = 0;
total = 0;
y_predict = zeros(size(y));
confidence = zeros(size(y));
for i = 1:length(y)
    num = exp(theta*x(:,i));
    temp_predict = num/sum(num);
    [confidence(i), y_predict(i)] = max(temp_predict);

    %for comparing numbered labels
    if (y(i) == y_predict(i))
        correct = correct + 1;
    end
    total = total + 1;
end
percentCorrect = (correct/total)*100;
end