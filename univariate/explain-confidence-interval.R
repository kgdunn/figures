
N = 5

base.sd = 50
base.mean = 700
steps <- seq(-3.5*base.sd, 3.5*base.sd, base.sd/100)

steps <- steps + base.mean

raw <- dnorm(steps, mean=base.mean, sd=base.sd)
raw.sample <- dnorm(steps, mean=base.mean, sd=base.sd/sqrt(N))

bitmap('explain-confidence-interval.png', type="png256", width=10, height=7, res=300, pointsize=14)

plot(steps, raw.sample, lwd=4, type="l", xlab=expression(paste("x (thin), or  ", bar(x), " (thick)")), ylab="Probability density", 
    cex.lab=1.5, cex.main=1.3, cex.sub=1.8, cex.axis=1.8,
    main="Raw data distribution, and sample mean distribution")
lines(steps, raw)
abline(v=base.mean)
legend(x=-3.2*base.sd + base.mean, y=0.015, legend=c("Sample mean's distribution", "Raw data's distribution"), lwd=c(4, 1), cex=1)
text(x=-2*base.sd + base.mean, 0.01,"Sample size, n = 5")
dev.off()


N = 5
base.sd = 1
base.mean = 0
steps <- seq(-3.5*base.sd, 3.5*base.sd, base.sd/100)
steps <- steps + base.mean
raw <- dnorm(steps, mean=base.mean, sd=base.sd)
raw.sample <- dnorm(steps, mean=base.mean, sd=base.sd/sqrt(N))

bitmap('explain-confidence-interval-normalized.png', type="png256", width=10, height=7, res=300, pointsize=14)

plot(steps, raw.sample, lwd=4, type="l", xlab="z", ylab="Probability density", 
    cex.lab=1.5, cex.main=1.3, cex.sub=1.8, cex.axis=1.8,
    ylim=c(0,1),
    main="Raw data distribution, and sample mean distribution")
lines(steps, raw)
abline(v=base.mean)
abline(h=0)
legend(x=-3.6*base.sd + base.mean, y=0.8, legend=c("Sample mean's distribution", "Raw data's distribution"), lwd=c(4, 1), cex=.7)
text(x=-2*base.sd + base.mean, 0.4,"Sample size, n = 5")

lower <- -qnorm(0.025)/sqrt(N)
upper <- -lower
abline(v=c(lower, upper), col="red", lty=2)
yval = 0.92
arrows(x0=lower, y0=yval, x1=upper, y1=yval, code=3, lwd=4, angle=15)
text(0, 0.95, "95% area", font=2)
dev.off()