function [J, grad] = costFunction(theta, X, y)

m = length(y);

z = X * theta;

g = sigmoid(z);

J  = 1 / m * (-y' * log(g) - (1 - y)' * log(1 - g));

difference = g - y;

grad = 1 / m * X' * difference;

end
