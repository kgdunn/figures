dof <- c(1, 2, 3, 4, 5, 10, 15, 20, 30, 60, Inf)
tail.area.oneside <- c(0.4, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005)

n.dof <- length(dof)
n.tails <- length(tail.area.oneside)

values <- matrix(0, nrow=n.dof, ncol=n.tails)
k=0
for (entry in tail.area.oneside){
    k=k+1
    values[,k] <- abs(qt(entry, dof))
}
round(values,3)

#        [,1]  [,2]  [,3]  [,4]   [,5]   [,6]   [,7]
#  [1,] 0.325 1.000 3.078 6.314 12.706 31.821 63.657
#  [2,] 0.289 0.816 1.886 2.920  4.303  6.965  9.925
#  [3,] 0.277 0.765 1.638 2.353  3.182  4.541  5.841
#  [4,] 0.271 0.741 1.533 2.132  2.776  3.747  4.604
#  [5,] 0.267 0.727 1.476 2.015  2.571  3.365  4.032
#  [6,] 0.260 0.700 1.372 1.812  2.228  2.764  3.169
#  [7,] 0.258 0.691 1.341 1.753  2.131  2.602  2.947
#  [8,] 0.257 0.687 1.325 1.725  2.086  2.528  2.845
#  [9,] 0.256 0.683 1.310 1.697  2.042  2.457  2.750
# [10,] 0.254 0.679 1.296 1.671  2.000  2.390  2.660
# [11,] 0.253 0.674 1.282 1.645  1.960  2.326  2.576
library(RSvgDevice)
devSVG("t-distribution-raw.svg", width=10, height=10)
par(mar=c(4.2, 4.2, 0.2, 1))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
z <- seq(-5, 5, 0.01)
probabilty <- dt(z, df=5)
plot(z, probabilty, type="l", main="", xlab="z", ylab="Probabilities from the t-distribution", 
                    cex.lab=1.4, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8)
abline(h=0)
z=1.5
abline(v=z)
abline(v=0)

dev.off()