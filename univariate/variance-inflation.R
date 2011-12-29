
nsim <- 1000            # Number of simulations
x.mean <- numeric(nsim) # An empty vector to store the results

set.seed(37)            # so that you can reproduce these results
for (i in 1:nsim)
{
    N <- 100            # number of points in autocorrelated sequence
    phi <- +0.7         # ** change this line for case A, B and C **
    spread <- 5.0       # standard deviation of random variables
    x <- numeric(N)
    x[1] = rnorm(1, mean=0, sd=spread)
    for (k in 2:N){
       x[k] <- phi*x[k-1] + rnorm(1, mean=0, sd=spread)
    }
    x.mean[i] <- mean(x)
}
theoretical <- sqrt(spread^2/N)

# Show some output to the user
c(theoretical, mean(x.mean), sd(x.mean))
