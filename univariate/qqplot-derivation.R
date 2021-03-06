N = 10
index <- seq(1, N)
P <- (index - 0.5) / N
theoretical.quantity <- qnorm(P)

yields = c(86.2, 85.7, 71.9, 95.3, 77.1, 71.4, 68.9, 78.9, 86.9, 78.4)
mean.yield = mean(yields)		# 80.0
sd.yield = sd(yields)			# 8.35
yields.z = (yields - mean.yield)/sd.yield
yields.z.sorted = sort(yields.z)

bitmap('qqplot-derivation.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.2, 0.2))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
plot(theoretical.quantity, yields.z.sorted, type="p", 
    cex.lab=1.5, cex.main=1.8, lwd=4, cex.sub=1.8, cex.axis=1.8,
    xlim=c(-1.8, 1.8), ylim=c(-1.8, 1.8))
dev.off()

bitmap('qqplot-from-R.png', type="png256", width=14, height=7, res=300, pointsize=14)

m <- t(matrix(seq(1,2), 2, 1))
layout(m)

qqnorm(yields, main="Using the built-in functions")
qqline(yields)

library(car)
qqPlot(yields, main="Using the car library", cex=1.5, lwd=2)

dev.off()