bitmap('t-distribution-comparison.png', type="png256", width=14, height=7, 
       res=250, pointsize=14)

x <- seq(-5, 5, 0.01)
p.x <- dnorm(x)
p.t <- dt(x, df=8)
plot(x, p.x, type="l", xlab="z", ylab="p(z)", frame.plot=FALSE, main="", 
     cex.lab=1.8, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8, xlim=c(-4,4))
lines(x, p.t, col="blue")
abline(v=0)

legend(1, y=0.35, legend=c("Normal distribution", "t-distribution (df=8)"), 
       lwd = c(4, 1), col=c("black", "blue"))

dev.off()

bitmap('t-distribution-comparison-3.png', type="png256", width=14, height=7, 
       res=250, pointsize=14)

x <- seq(-5, 5, 0.01)
p.x <- dnorm(x)
p.t <- dt(x, df=3)
plot(x, p.x, type="l", xlab="z", ylab="p(z)", frame.plot=FALSE, main="", 
     cex.lab=1.8, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8, xlim=c(-4,4))
lines(x, p.t, col="blue")
abline(v=0)

legend(1, y=0.35, legend=c("Normal distribution", "t-distribution (df=3)"), 
       lwd = c(4, 1), col=c("black", "blue"))

dev.off()

bitmap('t-distribution-comparison-5.png', type="png256", width=14, height=7, 
       res=250, pointsize=14)

x <- seq(-5, 5, 0.01)
p.x <- dnorm(x)
p.t <- dt(x, df=5)
plot(x, p.x, type="l", xlab="z", ylab="p(z)", frame.plot=FALSE, main="", 
     cex.lab=1.8, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8, xlim=c(-4,4), col=c("black", "blue"))
lines(x, p.t, col="blue")
abline(v=0)

legend(1, y=0.35, legend=c("Normal distribution", "t-distribution (df=5)"), 
       lwd = c(4, 1), col=c("black", "blue"))

dev.off()

bitmap('t-distribution-comparison-7.png', type="png256", width=14, height=7, 
       res=250, pointsize=14)

x <- seq(-5, 5, 0.01)
p.x <- dnorm(x)
p.t <- dt(x, df=7)
plot(x, p.x, type="l", xlab="z", ylab="p(z)", frame.plot=FALSE, main="", 
     cex.lab=1.8, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8, xlim=c(-4,4))
lines(x, p.t, col="blue")
abline(v=0)

legend(1, y=0.35, legend=c("Normal distribution", "t-distribution (df=7)"), 
       lwd = c(4, 1), col=c("black", "blue"))

dev.off()

bitmap('t-distribution-comparison-13.png', type="png256", width=14, height=7, 
       res=250, pointsize=14)

x <- seq(-5, 5, 0.01)
p.x <- dnorm(x)
p.t <- dt(x, df=13)
plot(x, p.x, type="l", xlab="z", ylab="p(z)", frame.plot=FALSE, main="", 
     cex.lab=1.8, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8, xlim=c(-4,4))
lines(x, p.t, col="blue")
abline(v=0)

legend(1, y=0.35, legend=c("Normal distribution", "t-distribution (df=13)"), 
       lwd = c(4, 1), col=c("black", "blue"))

dev.off()

bitmap('t-distribution-comparison-20.png', type="png256", width=14, height=7, 
       res=250, pointsize=14)

x <- seq(-5, 5, 0.01)
p.x <- dnorm(x)
p.t <- dt(x, df=20)
plot(x, p.x, type="l", xlab="z", ylab="p(z)", frame.plot=FALSE, main="", 
     cex.lab=1.8, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8, xlim=c(-4,4))
lines(x, p.t, col="blue")
abline(v=0)

legend(1, y=0.35, legend=c("Normal distribution", "t-distribution (df=20)"), 
       lwd = c(4, 1), col=c("black", "blue"))

dev.off()