lactose.raw <- c(90, 94, 90, 94, 88, 96, 92, 92, 92, 92, 92, 92)
glucose.raw <- c(85, 85, 90, 90, 87.5, 87.5, 82.5, 92.5, 87.5, 87.5, 87.5, 87.5)
yield.raw <- c(51, 79, 72, 94, 50, 99, 49, 89, 73, 74, 64, 65)

n <- length(lactose.raw)
 
x1 <- lactose.raw - mean(lactose.raw)
x2 <- glucose.raw - mean(glucose.raw)
y <- yield.raw - mean(yield.raw)
X <- cbind(x1, x2)

 
# Calculate:  b = inv(X'X) X'y
XTX <- t(X) %*% X    # compare this to cov(X)*(n-1)
XTY <- t(X) %*% y
XTX.inv <- solve(XTX)
b <- XTX.inv %*% XTY
b

model <- lm(yield.raw ~ lactose.raw + glucose.raw)
summary(model)
confint(model)

library(car)
bitmap('lactose-glucose-yield-residuals.png', type="png256", width=8, 
       height=8, res=300, pointsize=14)
qqPlot(resid(model))
dev.off()