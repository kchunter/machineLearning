function p = predict(theta, X)

m = size(X, 1);

p = zeros(m, 1);

z = X * theta;

g = sigmoid(z);

p(g>=0.5) = 1;

end
