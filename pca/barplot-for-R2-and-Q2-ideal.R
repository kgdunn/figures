
# Data from the Jurong data set (Dora Kourti) 
# Simca-P v 11.5 (Nov 2006)

R2 = c(0.286689,0.491033,0.566433,0.627405,0.677535,0.717479,0.756406,0.793391,0.826731,0.857203,0.88421)
Q2_simca = c(0.226146,0.413826,0.446406,0.476091,0.483116,0.472488,0.431964,0.403468,0.352738,0.288011,0.238454)


bitmap('barplot-for-R2-and-Q2-Simca-ideal.png', type="png256", width=8, height=3.0, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.5, 4.2, 1.0, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
layout(matrix(c(1,2), 1, 2))

barplot(R2, names.arg=seq(1,11), col=0, width=1, ylab=expression(paste("Cumulative ",  R^2, " "  )), ylim=c(0,1))
abline(h=c(0,1), col="black")

barplot(Q2_simca, names.arg=seq(1,11), col=0, width=1, ylab=expression(paste("Cumulative ",  Q^2, " "  )), ylim=c(0,1), main="Simca-P 11.5")
abline(v=3.7, col="darkred", lty=2)
abline(v=7.3, col="darkred", lty=2)
abline(h=c(0,1), col="black")
dev.off()
