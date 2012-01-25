steps <- seq(-6, 6, 0.01)
Delta = 1
N = 5

base.sd = 2
base.mean = 6
steps <- steps + base.mean

raw <- dnorm(steps, mean=base.mean, sd=base.sd)
raw.sample <- dnorm(steps, mean=base.mean, sd=base.sd/sqrt(N))


bitmap('explain-shewhart.png', type="png256", width=10, height=7, res=300, pointsize=14)

plot(steps, raw.sample, lwd=4, type="l", xlab="x", ylab="Probability density", 
    cex.lab=1.5, cex.main=1.3, cex.sub=1.8, cex.axis=1.8, xlim=c(0, 12),
    main="Shewhart chart: using theoretical (usually unknown) parameters")
lines(steps, raw)
abline(v=base.mean)

legend(x=-0.4, y=0.36, xjust=0, lwd=c(4, 1), cex=0.7,
      legend=c("Sub group data's distribution", (expression("Raw data's distribution: "*sigma*"=2")) ) )

#yval = dnorm(1)
#arrows(x0=base.mean, y0=yval, x1=base.mean+base.sd, y1=yval, code=2)
#abline(v=base.mean+base.sd)

text(1, 0.18,"Sub group size, n = 5", pos=1)
#text(10.9, 0.05, (expression(""*sigma*"=2")), cex=1.3)
text(7.9, 0.23, (expression(""*sigma[bar(x)]*" = 2/"*sqrt(5))), cex=1.3)

yval = 0.35
arrows(x0=base.mean, y0=yval, x1=base.mean+3*(base.sd/sqrt(N)), y1=yval, code=2, angle=15)
abline(v=base.mean+3*(base.sd/sqrt(N)), col="red", lwd=2)
abline(v=base.mean-3*(base.sd/sqrt(N)), col="red", lwd=2)
text(base.mean+1.6*(base.sd/sqrt(N)), yval-0.015, (expression("3"*sigma[bar(x)])), cex=1.3)

yval = 0.122
arrows(x0=base.mean, y0=yval, x1=base.mean+base.sd, y1=yval, code=2, angle=15)
text(base.mean+0.5*base.sd, yval-0.015, (expression(sigma)), cex=1.3)

dev.off()