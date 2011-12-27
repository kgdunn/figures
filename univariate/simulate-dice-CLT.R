N <- 10
n <- 10000
x.mean <- numeric(N)
x.var <- numeric(N)
for (i in 1:N) {
  x.data <- as.integer(runif(n, 1, 13))
  x.mean[i] <- mean(x.data)
  x.var[i] <- var(x.data)
}

x.mean
# [1] 6.5527 6.4148 6.4759 6.4967 6.4465 
# [6] 6.5062 6.5171 6.4671 6.5715 6.5485

x.var
# [1] 11.86561 11.84353 12.00102 11.89658 11.82552 
# [6] 11.83147 11.95224 11.88555 11.81589 11.73869

# You should run the code several times and verify whether
# the following values are around their expected, theoretical
# levels.  Some runs should be above, and other runs below 
# the theoretical values.  
# This is the same as increasing "N" in the first line.

# Is it around 6.5?
mean(x.mean)

# Is it around 11.9167?
mean(x.var)

# Is it around \sigma^2 / n = 11.9167/10000 = 0.00119167 ?
var(x.mean)