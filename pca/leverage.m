X = csvread('/Users/kevindunn/ChE-765/materials-for-class/class-4/silicon-wafer-thickness.csv', 1);

centerX = mean(X);
scaleX = std(X); 

[N, K] = size(X);
X = X - repmat(centerX, N, 1);
X = X ./ repmat(scaleX, N, 1);

[evec, eval] = eig(X'*X);

[eval_s, idx] = sort(diag(eval), 'descend');
P = evec(:, idx);
A = 3;
P = P(:, 1:A);
T = X * P;
s = std(T);



% Proves that leverage is just a scaled version of T^2
HotT2 = sum(T.^2 ./ repmat(s.^2, N, 1), 2);
Leverage = diag(T * inv(T'*T) * T');
plot(Leverage , 'ko-')
hold on
sum(Leverage)
cutoff = 3 * A / N;
plot([0, N], [cutoff cutoff], 'r-')
grid()
title(['Leverage points for wafer thickness data, using A = ', num2str(A)])

