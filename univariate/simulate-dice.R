# Set the random seed to a known point, to allow
# us to duplicate pseudorandom results
set.seed(13)

x.data <- as.integer(runif(10000, 1, 13))

# Verify that it is roughly uniformly distributed
# across 12 bins
hist(x.data, breaks=seq(0,12))

x.mean <- mean(x.data)
x.var <- var(x.data)
c(x.mean, x.var)