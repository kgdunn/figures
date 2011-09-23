
# Data from the LDPE case study; PCA on all the variables, all the rows; 
# Simca-P v 11.5 (Nov 2006)
# ProSensus Multivariate Revision 995 (Help/About), version 11.08, 2011

R2 = c(0.369877,0.547855,0.65697,0.747528,0.831526,0.885739,0.928337,0.962801, 0.990961, 0.99925, 0.999847)
Q2_simca = c(0.253787,0.341342,0.318556,0.250411,0.32386,0.270315,0.19828,0.245897, 0.774703, 0.963812, 0.990275)
Q2_pmv = c(0.32847, 0.50085, 0.575319, 0.6338, 0.7637, 0.8231, 0.87225, 0.916889, 0.986077, 0.998795, 0.99972)

bitmap('barplot-for-R2-and-Q2-Simca.png', type="png256", width=8, height=3.0, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.5, 4.2, 1.0, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
layout(matrix(c(1,2), 1, 2))

barplot(R2, names.arg=seq(1,11), col=0, width=1, ylab=expression(paste("Cumulative ",  R^2, " "  )), ylim=c(0,1))
abline(h=c(0,1), col="black")

barplot(Q2_simca, names.arg=seq(1,11), col=0, width=1, ylab=expression(paste("Cumulative ",  Q^2, " "  )), ylim=c(0,1), main="Simca-P 11.5")
abline(v=2.5, col="darkred", lty=2)
abline(h=c(0,1), col="black")
dev.off()

bitmap('barplot-for-R2-and-Q2-ProSensus.png', type="png256", width=8, height=3.0, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.5, 4.2, 1.0, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
layout(matrix(c(1,2), 1, 2))

barplot(R2, names.arg=seq(1,11), col=0, width=1, ylab=expression(paste("Cumulative ",  R^2, " "  )), ylim=c(0,1))
abline(h=c(0,1), col="black")

barplot(Q2_pmv, names.arg=seq(1,11), col=0, width=1, ylab=expression(paste("Cumulative ",  Q^2, " "  )), ylim=c(0,1), main="ProSensus 11.08")
abline(v=8.5, col="darkred", lty=2)
abline(h=c(0,1), col="black")
dev.off()