# Manually solve the least-squares model in R

x1 <- c(-4.5,  -2.5,  -1.5,  1.5,  3.5,  3.5)
x2 <- c(   4,     4,     1,   -2,   -4,   -3)
y <- matrix(c(-3.5,  -1.5,  -0.5,  1.5,  0.5,  3.5), nrow=6, ncol=1)
X <- cbind(x1, x2)

XtX <- t(X) %*% X
Xty <- t(X) %*% y

XtX.inv <- solve(XtX)

b <- XtX.inv %*% Xty

summary(lm(y ~ x1 + x2))


