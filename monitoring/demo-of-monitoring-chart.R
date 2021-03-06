# Use the same settings as "explain-CUSUM.R"
set.seed(42)
steps <- seq(0, 200)
N <- length(steps)
base.sd <- 3
base.mean <- 10
target <- base.mean
data <- rnorm(N, base.mean, base.sd)

step.point <- 150
step.fraction <- 1
data[step.point:N] <- data[step.point:N] + step.fraction*base.sd
ewma <- function(x, lambda, target=x[1]){
  N <- length(x)
  y <- numeric(N)
  y[1] = target
  for (k in 2:N)
  {
    error = x[k-1] - y[k-1]
    y[k] = y[k-1] + lambda*error
  }
  return(y)
}

bitmap('demo-of-monitoring-chart.png', type="png256", 
       width=12, height=4, res=300, pointsize=14)


lambda=0.4
LCL = base.mean-3*base.sd * sqrt(lambda/(2-lambda))
UCL = base.mean+3*base.sd * sqrt(lambda/(2-lambda))
plot( ewma(data, lambda=lambda, target=base.mean), 
      type="l", 
      ylim=c(LCL*0.90, UCL*1.15), 
      xlab="Time sequence order", 
      ylab="TC241 [degC]", 
      main="Tank temperature, TC241 [degC]", 
      cex.lab=1.5, 
      cex.main=1.2, 
      cex.sub=1.8, 
      cex.axis=1.8)
abline(h=base.mean, col="grey60", lwd=3)
lines(ewma(data, lambda=lambda, target=base.mean))
abline(h=LCL, col="red", lwd=2)
abline(h=UCL, col="red", lwd=2)

text(3, 15.5, "UCL", cex=1.2)
text(3, 6.5, "LCL", cex=1.2)

dev.off()