grades <- read.csv("http://openmv.net/file/unlimited-time-test.csv")
summary(grades)
y <- grades$Grade
x <- grades$Time

bitmap('unlimited-time-raw-data.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.8, cex.sub=1.8, cex.axis=1.5)
plot(x, y, xlab="Time [min]", ylab="Grade [%]")
dev.off()

model <- lm(y ~ x)
b0 <- model$coefficients[1]
b1 <- model$coefficients[2]
e <- resid(model)
y.hat <- b0 + b1*x
library(car)  # Plots to determine validity of model

bitmap('unlimited-time-residual-qq-plot.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.8, cex.sub=1.8, cex.axis=1.5)
qqPlot(e)     # Are the residuals normally distributed?
dev.off()

plot(x, e)    # Is there a correlation between to the residuals?
abline(h=0, lty=2)

# Calculate standard errors with subsets of n=14
n = length(x)
den <- sum((x - mean(x)) * (x - mean(x)))
SE <- sqrt(sum(e^2) / (n-2))
SE.b1 <- sqrt(SE^2 / den)

# Confidence intervals for the least squares parameters
alpha = 0.95
b1 = coef(model)[2]
t.critical = qt(1-(1-alpha)/2, df=(n-2))
b1.LB <- b1 - t.critical*SE.b1
b1.UB <- b1 + t.critical*SE.b1

ls.fun <- function(data, i){
  # `i` contains the indices of the rows to use in the
  # least squares model
  print (i)
  d.subset <- data[i,]
  d.ols <- lm(d.subset$Grade ~ d.subset$Time)
  
  # Return just the slope coefficient
  return(coef(d.ols)[2])
}

set.seed(42)
library(boot)
grades.boot <- boot(grades, ls.fun, R=1000)

library(MASS)

bitmap('unlimited-time-bootstrap.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.8, cex.sub=1.8, cex.axis=1.5)
truehist(grades.boot$t, col=0, xlab="Slope coefficient", ylab="Count [R = 1000]")
lines(density(grades.boot$t))

ymax=15
segments(x0=coef(model)[2], y0=0, y1 = ymax, col="blue", lwd=3)
segments(x0=b1.LB, y0=0, y1 = ymax, col="red", lwd=2)
segments(x0=b1.UB, y0=0, y1 = ymax, col="red", lwd=2)
arrows(x0=b1.LB, x1=b1.UB, y0=3, y1=3, angle=20, length=0.2, col="red", code=3)
text(x=(b1.LB+b1.UB)/2, y=3, "Actual confidence interval", pos=1, col="red")

delta=0.01
text(x=b1.LB-delta, y=(ymax-5), "CI: lower bound", srt=90, pos=4)
text(x=b1.UB-delta, y=(ymax-5), "CI: upper bound", srt=90, pos=4)
dev.off()