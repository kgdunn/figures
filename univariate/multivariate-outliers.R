set.seed(4)
N = 25
x <- rnorm(N, sd=5, mean=10)
y <- x*4 - 6 + log(x) +rnorm(N, sd=3)

x[N+1] = 12
y[N+1] = 70

model1 <- lm(y~x)

bitmap('multivariate-outliers.png', type="png256", width=7, 
       height=7, res=300, pointsize=14)
par(mar=c(4.2, 5, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.7, cex.main=1.8, cex.sub=1.8, cex.axis=1.8)
plot(x[1:N], y[1:N], cex=1.5, lwd=2, xlab="x", ylab="y", 
     xlim=c(min(x), max(x)+1), main="")
points(x[N+1], y[N+1], pch=22, cex=1.5, lwd=4)

abline(coef(model1), lwd=2, lty=1)
#abline(coef(model.rm), lwd=2, lty=2)
#legend(x=10, y=20, legend=c("Model using all data", "Model omitting square point"), col=c("black", "black"), lty=c(1, 2), lwd=c(2, 2))
dev.off()