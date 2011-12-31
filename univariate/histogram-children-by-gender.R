N = 2739
set.seed(7)
data <- cut (runif(N), 2)
summary(data)

bitmap('histogram-children-by-gender.png', type="png256", 
    width=7, height=7, res=300, pointsize=14)
par(mar=c(5, 4.2, 4, 2) + 0.1)  # (B, L, T, R); 
plot(data, xlab="Children born in Hamilton, April 2009, by gender", ylab= paste("Number of children (N=", N, ")"),  
    names.arg=c("Male", "Female"), ylim=c(0, N/2+50), cex.axis=1.5, cex.lab=1.5, cex.names=1.5, col="white")
dev.off()