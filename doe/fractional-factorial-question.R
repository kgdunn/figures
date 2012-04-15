A <- B <- C <- c(-1, 1)
d <- expand.grid(A=A, B=B, C=C)
y <- c(14.0, 16.8, 15.0, 15.4, 27.6, 24.0, 27.4, 22.6)
A <- d$A
B <- d$B
C <- d$C
D <- A*B
E <- A*C
F <- B*C
G <- A*B*C
model <- lm(y ~ A + B + C + D + E + F + G)
summary(model)

coeff <- coef(model)[2:length(coef(model))]
 
# Pareto plot of the absolute coefficients
library(lattice)
bitmap('fractional-factorial-question.png', type="png256",
        width=6, height=6, res=300, pointsize=14)
barchart(sort(abs(coeff)), xlab="Magnitude of effect", 
         ylab = "Effect", col=0)
dev.off()