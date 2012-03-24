A <- B <- C <- c(-1, 1)
grd <- expand.grid(A=A, B=B, C=C)
A <- grd$A
B <- grd$B
C <- grd$C
y <- c(0.805, 0.17, 0.79, 0.225, 0.765, 0.265, 0.765, 0.12)

mod.full <- lm(y ~ (A + B + C)^3)
summary(mod.full)
