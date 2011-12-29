V <- c(0.01, 0.12, 0.24, 0.38, 0.51, 0.67, 0.84, 1.01, 1.15, 1.31)
T <- c(273, 293, 313, 333, 353, 373, 393, 413, 433, 453)
model <- lm(T ~ V)
summary(model)

# ANOVA values
y.mean <- mean(T)
RegSS <- sum((predict(model) - y.mean)^2)
RSS <- sum(residuals(model)^2)
TSS <- sum((T - y.mean)^2)
mean.square.residual <- RSS / model$df.residual
c(RegSS, RSS, TSS, mean.square.residual)