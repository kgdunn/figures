data <- read.csv('http://openmv.net/file/gas-furnace.csv')
CO2 <- data$CO2
N.raw <- length(CO2)
N.sub <- 6  

# Change ``N.sub`` to 10, 15, 20, etc
# At N.sub <- 17 we see the autocorrelation disappear

# Plot all the data
bitmap('CO2-raw-data.png', type="png256", 
       width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5))  
par(cex.lab=1.3, cex.main=1.5, cex.sub=1.5, cex.axis=1.5)
plot(CO2, type="p", pch=".", cex=2, main="", 
     ylab="CO2: raw data", xlab="Sequence order")
dev.off()

# Create the subgroups on ALL the raw data.  Form a matrix with 
# `N.subgroup` rows by placing the vector of data down each row, 
# then going across to form the columns.

# Calculate the mean and standard deviation within each subgroup 
# (columns of the matrix)
subgroups <- matrix(CO2, N.sub, N.raw/N.sub)
subgroups.S <- apply(subgroups, 2, sd)
subgroups.xbar <- apply(subgroups, 2, mean)
ylim <- range(subgroups.xbar) + c(-3, +3)

# Keep adjusting N.sub until you don't see any autocorrelation 
# between subgroups
acf(subgroups.xbar)

# Create a function to calculate Shewhart chart limits
shewhart_limits <- function(xbar, S, sub.size, N.stdev=3){
    # Give the xbar and S vector containing the subgroup means 
    # and standard deviations.  Also give the subgroup size used.  
    # Returns the lower and upper control limits for the Shewhart 
    # chart (UCL and LCL) which are N.stdev away from the target.
    
    x.double.bar <- mean(xbar)     
    s.bar <- mean(S)
    an <- sqrt(2)*gamma(sub.size/2)/(sqrt(sub.size-1)*gamma((sub.size-1)/2))
    LCL <- x.double.bar - 3*s.bar/an/sqrt(sub.size)
    UCL <- x.double.bar + 3*s.bar/an/sqrt(sub.size)
return(list(LCL, x.double.bar, UCL))
}

limits <- shewhart_limits(subgroups.xbar, subgroups.S, N.sub)
LCL <- limits[1]
xdb <- limits[2]
UCL <- limits[3]
c(LCL, xdb, UCL)

# Any points outside these limits?  
bitmap('CO2-phaseI-first-round.png', type="png256",
       width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5))  
par(cex.lab=1.3, cex.main=1.5, cex.sub=1.5, cex.axis=1.5)
plot(subgroups.xbar, type="b", pch=".", cex=5, main="", ylim=ylim,
    ylab="Phase I subgroups: round 1",  xlab="Sequence order")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
abline(h=xdb, col="green")
lines(subgroups.xbar, type="b", pch=".", cex=5)
dev.off()