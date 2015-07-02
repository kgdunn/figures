data <- read.csv('http://openmv.net/file/flotation-cell.csv')
summary(data)

# Time-series data for the entire sequence of observations for the 
# ``Feed rate`` column.  Use the xts library for better plots; search the 
# software tutorial for "xts" to see how.
library(xts)
date.order <- as.POSIXct(data$Date.and.time, format="%d/%m/%Y %H:%M:%S") 
Feed.rate <- xts(data$Feed.rate, order.by=date.order)
bitmap('flotation-feedrate.png', res=300, pointsize=14, width=10, height=5)
par(mar=c(3.0, 4, 1, 0.2))
plot(Feed.rate, ylab="Feed rate", main="")
dev.off()

# Use all data on 15 December 2004 (points 1 to 479) from the ``Feed rate`` 
# variable as your phase 1 data. You have no other process information to go 
# on, so make any assumptions as required. Iteratively prune any outliers to
# settle on a reasonable set of monitoring parameters. A subgroup size of 4,
# representing 2 minutes of operation, should be used.
phase1.point <- 479
phase1 <- data$Feed.rate[1:phase1.point]

N.raw = length(phase1)
N.sub = 4                                              # subgroup size
subgroup.1 <- matrix(phase1, N.sub, N.raw/N.sub)
N.groups <- ncol(subgroup.1)
dim(subgroup.1)                                          # 4 by 119 matrix

subgroup.1.sd <- apply(subgroup.1, 2, sd)
subgroup.1.xbar <- apply(subgroup.1, 2, mean)

# Take a look at what these numbers mean
plot(subgroup.1.sd, type="b",  ylab="Subgroup spread")
# there's evidence process really isn't stable
plot(subgroup.1.xbar, type="b", ylab="Subgroup average") 

# Report your target value, lower control limit and upper control limit,
target <- mean(subgroup.1.xbar)
Sbar <- mean(subgroup.1.sd)
an <- sqrt(2) * gamma(N.sub/2) / (sqrt(N.sub-1) * gamma(N.sub/2 - 0.5))
sigma.estimate <- Sbar / an  # a_n value is from the table when subgroup size = 5
LCL <-  target - 3 * sigma.estimate/sqrt(N.sub)
UCL <- target + 3 * sigma.estimate/sqrt(N.sub)
c(LCL, target, UCL)

# Sequence order Shewhart-chart
bitmap('flotation-feedrate-phase1.png', res=300, pointsize=14, width=10, height=5)
par(mar=c(2, 4, 1, 0.2))
plot(subgroup.1.xbar, ylab="Feed rate subgroup averages", 
      main="Shewhart chart (phase 1)", ylim=c(LCL-5, UCL+5), type="b")
abline(h=target, col="green")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
dev.off()

# Improved Shewhart chart: uses time on the x-axis
Feed.rate.subgroup.1 <- xts(subgroup.1.xbar, 
           order.by=date.order[seq(N.sub, phase1.point, by=N.sub)])
bitmap('flotation-feedrate-phase1-time.png', res=300, pointsize=14, width=10, height=5)
par(mar=c(3, 4, 1, 0.2))
plot(Feed.rate.subgroup.1, ylab="Feed rate subgroup averages", 
      main="Shewhart chart (phase 1)", ylim=c(LCL-5, UCL+5), type="b")
abline(h=target, col="green")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
dev.off()

#.	Use data from 16 December 2004 (points 480 onwards) from the ``Feed rate``
# variable as your phase 2 (i.e. testing) data. Show the performance of 
# your monitoring parameters calculated in part 1 on these phase 2 data.
phase2.point <- length(data$Feed.rate)
phase2 <- data$Feed.rate[480:phase2.point]
N.raw = length(phase2)
N.sub = 4                                              # subgroup size
subgroup.2 <- matrix(phase2, N.sub, N.raw/N.sub)
N.groups <- ncol(subgroup.2)
dim(subgroup.2)                                        # 4 by 610 matrix

subgroup.2.sd <- apply(subgroup.2, 2, sd)
subgroup.2.xbar <- apply(subgroup.2, 2, mean)

# Take a look at what these numbers mean
plot(subgroup.2.sd, type="b",  ylab="Subgroup spread")
# there's evidence process really isn't stable
plot(subgroup.2.xbar, type="b", ylab="Subgroup average")

# Ordinary phase 2 Shewhart chart
bitmap('flotation-feedrate-phase2.png', type="png256", width=10, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.2, 0.2))  # (B, L, T, R); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
plot(subgroup.2.xbar, ylab="Subgroup means",
     main="Shewhart chart (phase 2)", ylim=c(250, 380), type="b")
abline(h=target, col="green")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
dev.off()

# Improved Shewhart chart: uses time on the x-axis
Feed.rate.subgroup.2 <- xts(subgroup.2.xbar, 
     order.by=date.order[seq(phase1.point+N.sub, phase2.point, by=N.sub)])
bitmap('flotation-feedrate-phase2-time.png', res=300, pointsize=14, width=10, height=5)
par(mar=c(3, 4, 1, 0.2))
plot(Feed.rate.subgroup.2, ylab="Feed rate subgroup averages", 
      main="Shewhart chart (phase 2)", ylim=c(250, 380), type="l")
abline(h=target, col="green")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
dev.off()

# Implement an alternative monitoring chart using these same data: EWMA chart.
# EWMA function below is directly from the course notes
ewma <- function(x, lambda, target=x[1]){
    N <- length(x)
    y <- numeric(N)
    y[1] = target
    for (k in 2:N){
        error = x[k-1] - y[k-1]
        y[k] = y[k-1] + lambda*error
    }
return(y)
}
# Use all data in the EWMA chart
N.raw = length(data$Feed.rate)
N.sub = 4                                              # subgroup size
subgroup <- matrix(data$Feed.rate, N.sub, N.raw/N.sub)
N.groups <- ncol(subgroup)
subgroup.xbar <- apply(subgroup, 2, mean)
lambda <- 0.15

# Calculate the EWMA values around the phase 1 target
ewma.values <- ewma(subgroup.xbar, lambda, target=mean(subgroup.1.xbar))
Feed.rate.ewma <- xts(ewma.values, order.by=date.order[seq(N.sub, N.raw, by=N.sub)])
LCL <-  target - 3 * sigma.estimate/sqrt(N.sub) * sqrt(lambda/(2-lambda))
UCL <- target + 3 * sigma.estimate/sqrt(N.sub) * sqrt(lambda/(2-lambda))

bitmap('flotation-feedrate-EWMA-time.png', res=300, pointsize=14, width=10, height=5)
par(mar=c(3, 4, 1, 0.2))
plot(Feed.rate.ewma, ylab="Feed rate subgroup averages", 
      main="EWMA chart (lambda = 0.15)", ylim=c(310, 350), type="l")
abline(h=target, col="green")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
dev.off()
