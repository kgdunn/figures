cofa<-c(50.9, 52.9, 51.6, 50.8, 54.6, 52.9, 53.1, 48.4, 51.6, 53.1, 53.8, 52.4, 53.1, 50.8,54.6, 52.9, 50.0, 53.8, 54.6, 52.2)
median(cofa)
mean(cofa)
sd(cofa)
mad(cofa)

bitmap('Certificates-of-analysis-sequence.png', 
       type="png256", width=7, height=7, res=300, 
       pointsize=14)
par(mar=c(4, 4.2, 2, 0.2))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
plot(cofa, ylab="Particle size", xlab="Shipment number", cex=2)
grid()
dev.off()

library(car)
bitmap('Certificates-of-analysis-qqplot.png', 
       type="png256", width=7, height=7, res=300, 
       pointsize=14)
par(mar=c(4, 4.2, 2, 0.2))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
qqPlot(cofa, ylab="Particle size", cex=2)
dev.off()