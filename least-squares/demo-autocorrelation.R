set.seed(11)
x <- rnorm(1004,50,6)

lims = c(30,70)
bitmap('random-1-step-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
plot(x[1:1000], x[2:1001], asp=1, xlim=lims, ylim=lims)
model <- lm(x[2:1001] ~ x[1:1000])
abline(model, col="darkgreen", lwd=2)
text(30, 30, paste("Correlation = r = ", round(cor(x[2:1001], x[1:1000]), 2)), 
     col="darkgreen", cex=1.5, adj = c(0, NA))
dev.off()

bitmap('random-2-step-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
plot(x[1:1000], x[3:1002], asp=1, xlim=lims, ylim=lims)
model <- lm(x[3:1002] ~ x[1:1000])
abline(model, col="darkgreen", lwd=2)
text(30,30, paste("Correlation = r = ", round(cor(x[3:1002], x[1:1000]), 2)), 
     col="darkgreen", cex=1.5, adj = c(0, NA))
dev.off()

bitmap('random-3-step-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
plot(x[1:1000], x[4:1003], asp=1, xlim=lims, ylim=lims)
model <- lm(x[4:1003] ~ x[1:1000])
abline(model, col="darkgreen", lwd=2)
text(30, 30, paste("Correlation = r = ", round(cor(x[4:1003], x[1:1000]), 2)), 
     col="darkgreen", cex=1.5, adj = c(0, NA))
dev.off()

bitmap('random-4-step-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
plot(x[1:1000], x[5:1004], asp=1, xlim=lims, ylim=lims)
model <- lm(x[5:1004] ~ x[1:1000])
abline(model, col="darkgreen", lwd=2)
text(30, 30, paste("Correlation = r = ", round(cor(x[5:1004], x[1:1000]), 2)), 
     col="darkgreen", cex=1.5, adj = c(0, NA))
dev.off()

bitmap('random-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.5, cex.sub=1.5, cex.axis=1.5 )
acf(x, lag.max=14, ylab="Correlation values", main="Autocorrelation function", xlab="Lags")
text(0, 1.02, "1.0", col="darkgreen", cex=1)
text(1, 0.09, "+0.02", col="darkgreen", cex=1)
text(2, -0.04, "-0.03", col="darkgreen", cex=1)
text(3, 0.09, "+0.03", col="darkgreen", cex=1)
text(4, 0.045, "+0.02", col="darkgreen", cex=1)
dev.off()



N=1005
set.seed(22)
long <- numeric(N)
phi_long = 0.65
for (k in 2:N){
  long[k] = rnorm(1, sd=4) + phi_long*long[k-1]
}
x <- long + 50
summary(x)

bitmap('autocorr-1-step-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
plot(x[1:1000], x[2:1001], asp=1, xlim=lims, ylim=lims)
model <- lm(x[2:1001] ~ x[1:1000])
abline(model, col="darkgreen", lwd=2)
text(30, 30, paste("Correlation = r = ", round(cor(x[2:1001], x[1:1000]), 2)), 
     col="darkgreen", cex=1.5, adj = c(0, NA))
dev.off()

bitmap('autocorr-2-step-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
plot(x[1:1000], x[3:1002], asp=1, xlim=lims, ylim=lims)
model <- lm(x[3:1002] ~ x[1:1000])
abline(model, col="darkgreen", lwd=2)
text(30, 30, paste("Correlation = r = ", round(cor(x[3:1002], x[1:1000]), 2)), 
     col="darkgreen", cex=1.5, adj = c(0, NA))
dev.off()

bitmap('autocorr-3-step-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
plot(x[1:1000], x[4:1003], asp=1, xlim=lims, ylim=lims)
model <- lm(x[4:1003] ~ x[1:1000])
abline(model, col="darkgreen", lwd=2)
text(30, 30, paste("Correlation = r = ", round(cor(x[4:1003], x[1:1000]), 2)), 
     col="darkgreen", cex=1.5, adj = c(0, NA))
dev.off()

bitmap('autocorr-4-step-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.9, cex.sub=1.8, cex.axis=1.8 )
plot(x[1:1000], x[5:1004], asp=1, xlim=lims, ylim=lims)
model <- lm(x[5:1004] ~ x[1:1000])
abline(model, col="darkgreen", lwd=2)
text(30, 30, paste("Correlation = r = ", round(cor(x[5:1004], x[1:1000]), 2)), 
     col="darkgreen", cex=1.5, adj = c(0, NA))
dev.off()

bitmap('autocorr-acf.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.5, cex.sub=1.5, cex.axis=1.5 )
acf(x, lag.max=14, ylab="Correlation values", main="Autocorrelation function", xlab="Lags")
text(0, 1.02, "1.0", col="darkgreen", cex=1)
text(1, 0.635, "0.61", col="darkgreen", cex=1)
text(2, 0.375, "0.35", col="darkgreen", cex=1)
text(3, 0.21, "0.19", col="darkgreen", cex=1)
text(4, 0.13, "0.11", col="darkgreen", cex=1)
dev.off()

bitmap('autocorr-acf-subsampled.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.5, cex.sub=1.5, cex.axis=1.5 )
x.subsample <- x[seq(1, length(long), 4)]
acf(x.subsample, lag.max=14, ylab="Correlation values", main="Autocorrelation function", xlab="Lags")
text(0, 1.02, "1.0", col="darkgreen", cex=1)
dev.off()

# bitmap('series-original.png', type="png256", width=7, height=7, res=300, pointsize=14)
# par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
# par(cex.lab=1.5, cex.main=1.5, cex.sub=1.5, cex.axis=1.5 )
# plot(long, type="l")
# dev.off()
# 
# bitmap('series-subsampled.png', type="png256", width=7, height=7, res=300, pointsize=14)
# par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
# par(cex.lab=1.5, cex.main=1.5, cex.sub=1.5, cex.axis=1.5 )
# plot(x.subsample , type="l")
# dev.off()