N = 100
set.seed(11)
data <- as.numeric(cut (runif(N), breaks=c(0, 0.7, 1.0)))
data <- gl(2, 70, length=100, labels=c("Pass", "Fail"))

bitmap('histogram-70-30.png', type="png256", res=300, pointsize=14)
par(mar=c(2.2, 5.2, 2, 0.2))  # (B, L, T, R); par(mar=c(5, 4, 4, 2) + 0.1)
plot(data, xlab="", 
           ylab="Percentage of tablets",         
           names.arg=c("Pass", "Fail"), 
           ylim=c(0, 100),
           col="white",
           cex.lab=2, cex.main=2, lwd=2, cex.sub=2, cex.axis=2, cex.names=2
     )
dev.off()