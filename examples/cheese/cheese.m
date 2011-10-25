cheese_data = csvread('/Users/kevindunn/ConnectMV/Datasets/Cheese/cheddar-cheese.csv', 1, 1);

X = cheese_data(:, 1:3);
y = cheese_data(:, 4);
[N, K] = size(X);
X = X - repmat(mean(X), N, 1);

beta = inv(X'*X) * X'*y;
