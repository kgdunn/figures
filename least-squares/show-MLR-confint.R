set.seed(37)
N=10


volume <- round(rnorm(N, 10, 3))
temperature <- round(rnorm(N, 273+32,5))
y <- round(-0.5*volume + 4.2*temperature + rnorm(N,125) )

# ... code to load the data ...

model <- lm(y ~ volume + temperature)
summary(model)
confint(model)




















