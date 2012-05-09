set.seed(0)
x1 <- rnorm(10)
x2 <- x1 * 2 + rnorm(10, sd=0.01)
x3 <- rnorm(10)
y  <- 3*x1 + 4*x2 - 5*x3 + rnorm(10, sd=0.05)

data <- data.frame(x1, x2, x3, y)
round(data,2)
plot(data)
mod.corr <- lm(y ~ x1 + x2 + x3)
summary(mod.corr)
confint(mod.corr)

x1[1] = 1.25
mod.updated <- lm(y ~ x1 + x2 + x3)
summary(mod.updated)
confint(mod.updated)

