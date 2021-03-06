values = seq(-5, 5, 0.01)
p.t.df8 = dt(values, df=8)
bitmap('t-values-8-DOF.png', type="png256", res=300, width=9, pointsize=14)
plot(values, p.t.df8, type="l", xlab="x", ylab="p(x) for t-distribution: 8 DOF", cex.lab=1.5, cex.main=1.8, lwd=2, cex.sub=1.8, cex.axis=1.8)
dev.off()