N = 100
data <- cut (runif(N), breaks=c(0, 0.7, 1.0))

bitmap('histogram-70-30.png', type="png256", res=300, pointsize=14)
par(mar=c(2.2, 4.2, 2, 0.2))  # (B, L, T, R); par(mar=c(5, 4, 4, 2) + 0.1)
plot(data, xlab="", 
           ylab="Percentage of tablets", 
           names.arg=c("Acceptable", "Defective"), 
           ylim=c(0, 100),
           col="white"
     )
dev.off()