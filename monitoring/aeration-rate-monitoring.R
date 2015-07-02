# Code by Ryan and Stuart (2011 class)

CUSUM <- function(x, target){
    N <- length(x)
    S <- numeric(N)
    S[1] = x[1] - target
    for (t in 2:N){        
        S[t] = S[t-1] + (x[t] - target)
    }
return(S)
}

# Import data and remove missing values (NA)
aeration.data <- read.csv('http://openmv.net/file/aeration-rate.csv')
aeration <- na.omit(aeration.data$Aeration)

# Plot raw data
bitmap('aeration-rate-raw-data.png', type="png256", 
        width=10, height=4, res=300, pointsize=14)
plot(aeration, type="l", xlab="Time (min)", ylab="Aeration rate (L/min)")
grid()
dev.off()

# Plot CUSUM Chart
target <- median(aeration[1:200])
bitmap('aeration-CUSUM.png', type="png256", 
        width=10, height=4, res=300, pointsize=14)
plot(CUSUM(aeration, target), type="l", xlab="Time (min)", 
     ylab="CUSUM cumulative deviations")
grid()
dev.off()

# Plot the Shewhart chart: see code from the other question to 
# calculate the control limits
LCL <- 22.1
UCL <- 25.8
N <- 5
subgroups <- matrix(aeration, N, length(aeration)/N)
x.mean <- numeric(length(aeration)/N)
x.sd <- numeric(length(aeration)/N)

# Calculate mean and sd of subgroups (see R-tutorial)
x.mean <- apply(subgroups, 2, mean)
x.sd <- apply(subgroups, 2, sd)
ylim <- range(x.mean) + c(-5, +5)
xdb <- target  # use the same CUSUM target !

bitmap('aeration-Shewhart-chart.png',
       type="png256", width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5)) 
par(cex.lab=1.3, cex.main=1.5, cex.sub=1.5, cex.axis=1.5)
plot(seq(1,length(x.mean)*N, N), x.mean, type="b", pch=".", cex=5, main="", 
     ylab="Phase II subgroups", xlab="Time order", ylim=ylim)
abline(h=UCL, col="red")
abline(h=LCL, col="red")
abline(h=xdb, col="green")
lines(c(275, 275), ylim, col="blue")
text(280, 29, "CUSUM detected problem at t=275",adj = c(0,0))
dev.off()
