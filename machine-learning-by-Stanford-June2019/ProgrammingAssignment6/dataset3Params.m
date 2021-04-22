function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

% potential values of C and sigma
pot_values = [0.01 0.03 0.1 0.3 1 3 10 30];

minError = Inf;
minC = Inf;
minsigma = Inf;


for i=1:length(pot_values)		% for each C
	for j=1:length(pot_values)	 %for each sigma

		currentC = pot_values(i);
		currentsigma = pot_values(j);

		% compute model for the specific (i,j) set of C, sigma
		model = svmTrain( X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentsigma) );
		
		% compute predictions on cross validation for the (i,j)th model 
		predictions = svmPredict(model, Xval);

		% compute prediction error 
		prediction_error = mean(double(predictions ~= yval));
		

		% check if prediction_error is minimum and , if yes, take the current C and sigma
		if prediction_error < minError
			minC = currentC;
			minsigma = currentsigma;
			minError = prediction_error;
		end
		

	end
end 

% take minimum C and sigma
C = minC;
sigma = minsigma;
		
% =========================================================================

end
