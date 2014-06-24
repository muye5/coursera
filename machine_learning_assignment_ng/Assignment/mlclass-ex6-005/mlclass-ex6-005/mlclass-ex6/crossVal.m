%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0.0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
load('ex6data3.mat');
c_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
x1 = [1 2 1]; x2 = [0 4 -1];

pred_error = Inf;
for i = 1:length(c_vec)
  c = c_vec(i);
  for j = 1:length(sigma_vec)
    sig = sigma_vec(j);
    model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    if pred_error > err
      pred_error = err;
      C = c;
      sigma = sig;
    endif
  endfor
endfor

fprintf(['best C = %f and best sigma = %f\n'], C, sigma);





% =========================================================================
