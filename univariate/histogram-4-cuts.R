N = 1000
data <- cut (runif(N)*4-2, breaks=c(-2, -1, 0, +1, +2))

bitmap('histogram-4-cuts.png', type="png256", res=300, pointsize=14)
par(mar=c(2.2, 4.2, 2, 0.2))  # (B, L, T, R); par(mar=c(5, 4, 4, 2) + 0.1)
plot(data, xlab="Defect type", 
           ylab="Number of defects", 
           names.arg=c("A", "B", "C", "D"), 
           ylim=c(0, 0.3*N),
           col="white"
     )
dev.off()