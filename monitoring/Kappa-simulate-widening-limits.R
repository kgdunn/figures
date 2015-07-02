kappa <- read.csv('http://openmv.net/file/kappa-number.csv')
summary(kappa)
attach(kappa)    # gives access to the variable "Kappa"

N.all <- length(Kappa)
N.phase1 <- 2000
N.phase2 <- N.all - N.phase1
N.subgroup <- 5

phase1.start <- 1
phase1.end <- floor(N.phase1/N.subgroup)
phase2.start <- phase1.end + 1
phase2.end <- floor(N.all/N.subgroup)

# We won't check the phase I raw data for outliers; we will use the phase I 
# subgroups to check for outliers.

# Create the subgroups on ALL the raw data.  Form a matrix with `N.subgroup` rows
# placing the vector of data down each row, then going across to form the columns.

# Calculate the mean and standard deviation within each subgroup (columns of the matrix)
reshaped.data <- matrix(Kappa, N.subgroup, N.all/N.subgroup)
subgroup.x.bar <- apply(reshaped.data, 2, mean)
subgroup.S <- apply(reshaped.data, 2, sd)

phase1.xbar <- subgroup.x.bar[phase1.start:phase1.end]
phase1.S <- subgroup.S[phase1.start:phase1.end]
phase2.xbar <- subgroup.x.bar[phase2.start:phase2.end]
phase2.S <- subgroup.S[phase2.start:phase2.end]

# We are going to repeatedly have to calculate the phase 1 limits.  Create a function.
shewhart_limits <- function(xbar, S, subgroup.size, N.stdev=3){
  # Give the xbar and S vector containing the subgroup means and standard deviations.
  # Also give the subgroup size used.  Returns the lower and upper control limits
  # for the Shewhart chart (UCL and LCL) which are N.stdev away from the target.
  
  x.double.bar <- mean(xbar)     
  s.bar <- mean(S)
  an = c(NA, 0.793, 0.886, 0.921, 0.940, 0.952, 0.959, 0.965)
  LCL <- x.double.bar - 3*s.bar/an[subgroup.size]/sqrt(subgroup.size)
  UCL <- x.double.bar + 3*s.bar/an[subgroup.size]/sqrt(subgroup.size)
  c(LCL, UCL)
  
  return(list(LCL, x.double.bar, UCL))
}
limits <- shewhart_limits(phase1.xbar, phase1.S, N.subgroup)
LCL <- as.numeric(limits[1])
xdb <- limits[2]
UCL <- as.numeric(limits[3])
c(LCL, xdb, UCL)

outside <- (phase1.xbar > UCL) + (phase1.xbar < LCL)
error <- round((sum(outside)+1)/length(outside)*100,1)

# Any points outside these limits?  Yup, quite a few.
bitmap('Kappa-assume-testing-with-outliers-01.png', type="png256", width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.3, cex.main=1.3, cex.sub=1.5, cex.axis=1.5)
plot(phase1.xbar, type="b", pch=".", cex=5, main="Shewhart monitoring chart", ylab="Kappa number", 
     xlab="Sequence order")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
abline(h=xdb, col="darkgreen")
text(30, 27, paste("Type 1 error: ", error, "%"))
lines(phase1.xbar, type="b", pch=".", cex=5)
dev.off()



LCL <- LCL-0.5
UCL <- UCL+0.5
outside <- (phase1.xbar > UCL) + (phase1.xbar < LCL)
error <- round((sum(outside))/length(outside)*100,1)

bitmap('Kappa-assume-testing-with-outliers-02.png', type="png256", width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.3, cex.main=1.3, cex.sub=1.5, cex.axis=1.5)
plot(phase1.xbar, type="b", pch=".", cex=5, main="Shewhart monitoring chart", ylab="Kappa number", 
     xlab="Sequence order")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
abline(h=xdb, col="darkgreen")
text(30, 27, paste("Type 1 error: ", error, "%"))
lines(phase1.xbar, type="b", pch=".", cex=5)
dev.off()

LCL <- LCL-0.5
UCL <- UCL+0.5
outside <- (phase1.xbar > UCL) + (phase1.xbar < LCL)
error <- round((sum(outside))/length(outside)*100,1)

bitmap('Kappa-assume-testing-with-outliers-03.png', type="png256", width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.3, cex.main=1.3, cex.sub=1.5, cex.axis=1.5)
plot(phase1.xbar, type="b", pch=".", cex=5, main="Shewhart monitoring chart", ylab="Kappa number", 
     xlab="Sequence order")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
abline(h=xdb, col="darkgreen")
text(30, 27, paste("Type 1 error: ", error, "%"))
lines(phase1.xbar, type="b", pch=".", cex=5)
dev.off()

LCL <- LCL-0.5
UCL <- UCL+0.5
outside <- (phase1.xbar > UCL) + (phase1.xbar < LCL)
error <- round((sum(outside))/length(outside)*100,1)

bitmap('Kappa-assume-testing-with-outliers-04.png', type="png256", width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.3, cex.main=1.3, cex.sub=1.5, cex.axis=1.5)
plot(phase1.xbar, type="b", pch=".", cex=5, main="Shewhart monitoring chart", ylab="Kappa number", 
     xlab="Sequence order")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
abline(h=xdb, col="darkgreen")
text(30, 27, paste("Type 1 error: ", error, "%"))
lines(phase1.xbar, type="b", pch=".", cex=5)
dev.off()


LCL <- LCL-0.5
UCL <- UCL+0.5
outside <- (phase1.xbar > UCL) + (phase1.xbar < LCL)
error <- round((sum(outside))/length(outside)*100,1)

bitmap('Kappa-assume-testing-with-outliers-05.png', type="png256", width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.3, cex.main=1.3, cex.sub=1.5, cex.axis=1.5)
plot(phase1.xbar, type="b", pch=".", cex=5, main="Shewhart monitoring chart", ylab="Kappa number", 
     xlab="Sequence order")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
abline(h=xdb, col="darkgreen")
text(30, 27, paste("Type 1 error: ", error, "%"))
lines(phase1.xbar, type="b", pch=".", cex=5)
dev.off()

LCL <- LCL-0.5
UCL <- UCL+0.5
outside <- (phase1.xbar > UCL) + (phase1.xbar < LCL)
error <- round((sum(outside))/length(outside)*100,1)

bitmap('Kappa-assume-testing-with-outliers-06.png', type="png256", width=10, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.3, cex.main=1.3, cex.sub=1.5, cex.axis=1.5)
plot(phase1.xbar, type="b", pch=".", cex=5, main="Shewhart monitoring chart", ylab="Kappa number", 
     xlab="Sequence order")
abline(h=UCL, col="red")
abline(h=LCL, col="red")
abline(h=xdb, col="darkgreen")
text(30, 27, paste("Type 1 error: ", error, "%"))
lines(phase1.xbar, type="b", pch=".", cex=5)
dev.off()


