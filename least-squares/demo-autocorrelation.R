N=1000
set.seed(22)
long <- numeric(N)
phi_long = 0.65
for (k in 2:N){
  long[k] = rnorm(1, sd=3) + phi_long*long[k-1]
}
#bitmap('demonstrate-autocorrelation.png', type="png256", width=14, height=14/3*2, res=300, pointsize=14)
#par(mar=c(4.2, 4.2, 4.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
#par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
#layout(matrix(c(1,2,3,4, 5, 6 ), 2, 3))
acf(long, main=expression(paste("Long autocorrelation: ", phi, " = 0.80")), ylab="")
#par(mar=c(4.2, 4.2, 0.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
#plot(long, type='b', ylab="")
#abline(h=0, lty=2)


x.subsample <- long[seq(1, length(long), 4)]
acf(x.subsample)

set.seed(65)
x <- rnorm(1000,50,6)
acf(x)