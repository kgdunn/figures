
bitmap('show-pnorm-and-qnorm.png', type="png256", width=10, height=7, res=300, pointsize=14)
par(mar=c(4, 4.5, 0.2, 0.2))  # (B, L, T, R) par(mar=c(5, 4, 4, 2) + 0.1)

z <- seq(-4, 4, 0.01)
cumulative.norm = pnorm(z)

plot(z, cumulative.norm, type="l", main="", xlab="z", ylab="Cumulative area under normal distribution",  
    cex.lab=1.5, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8, ylim=c(0, 1))



a1 = -0.5
arrows(a1, y=-0.2, x1=a1, y1=pnorm(a1), code=0, lwd=2)
arrows(a1, y=pnorm(a1), x1=-4, y1=pnorm(a1), code=2, lwd=2)
text(-2, pnorm(a1)+0.05, "pnorm(...)", cex=1.5)


a1 = 1
arrows(a1, y=0, x1=a1, y1=pnorm(a1), code=1, lwd=2)
arrows(a1, y=pnorm(a1), x1=-5, y1=pnorm(a1), code=0, lwd=2)
text(2,0.6, "qnorm(...)", cex=1.5)

abline(v=0, lty=2)

dev.off()

bitmap('show-pnorm-and-qnorm-qqplot.png', type="png256", width=10, height=7, res=300, pointsize=14)
par(mar=c(4, 4.5, 0.2, 0.2))  # (B, L, T, R) par(mar=c(5, 4, 4, 2) + 0.1)

z <- seq(-4, 4, 0.01)
cumulative.norm = pnorm(z)

plot(z, cumulative.norm, type="l", main="", xlab="z", ylab="Cumulative area under normal distribution",  
     cex.lab=1.5, cex.main=1.8, lwd=5, cex.sub=1.8, cex.axis=1.8, ylim=c(0, 1),
     col="black")


a1 = 0.95
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")
text(-3,0.8, "qnorm(...)", cex=1.5)

a1 = 0.85
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

a1 = 0.75
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

a1 = 0.65
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

a1 = 0.55
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

a1 = 0.45
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

a1 = 0.35
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

a1 = 0.25
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

a1 = 0.15
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

a1 = 0.05
arrows(qnorm(a1), y=a1, x1=-5, y1=a1, code=0, lwd=2, col="blue")
arrows(qnorm(a1), y=0, x1=qnorm(a1), y1=a1, code=1, lwd=2, col="blue")

abline(v=0, lty=2)

dev.off()