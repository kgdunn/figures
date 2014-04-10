# Generate the design matrix
A <- B <- C <- D <- c(-1, 1)
f <- expand.grid(A=A, B=B, C=C, D=D)
A <- f$A
B <- f$B
C <- f$C
D <- f$D
# Set the response values in standard order, and solve thefull factorial model
y <- c(60, 59, 63, 61, 69, 61, 94, 93, 56, 63, 70, 65, 44, 45, 78, 77)

mod.full <- lm(y ~ (A+B+C+D)^4)
b <- coef(mod.full)

# Pareto plot
coeff.full <- coef(mod.full)[2:length(coef(mod.full))]
library(lattice)




bitmap('bioreactor-pareto-plot.png', type="png256", width=8, 
        height=8, res=300, pointsize=14)
library(lattice)
coeff <- sort(abs(coeff.full), index.return=TRUE)
barchart(coeff$x, 
         xlim=c(0, max(abs(coeff.full))+0.1),
         xlab=list("Magnitude of effect", cex=1.5), 
         ylab = list("Effect", cex=1.5),
         groups=(coeff.full>0)[coeff$ix], col=c("lightblue", "orange"),
         scales=list(cex=1.5)
)
dev.off()

# Refit the model with only: B, C, D, BC, CD and intercept
mod.partial <- lm(y ~ B + C + D + B*C + C*D)
summary(mod.partial) 
# and check confidence intervals
confint(mod.partial)

# Half-fraction generated from D = A*B*C defining relationship I = ABCD

# Create a logical vector, indicating which subset of the full runs to use:
subset <- D == A*B*C  
A.s <- A[subset]
B.s <- B[subset]
C.s <- C[subset]
D.s <- D[subset]

y[subset]
 
mod.frac <- lm(y[subset] ~ A.s + B.s + C.s + D.s + A.s*B.s + A.s*C.s + A.s*D.s)
summary(mod.frac)
coeff.frac <- coef(mod.frac)[2:length(coef(mod.frac))]

bitmap('bioreactor-pareto-plot-half-fraction.png', type="png256", width=8, 
       height=8, res=300, pointsize=14)

coeff <- sort(abs(coeff.frac), index.return=TRUE)
barchart(coeff$x, 
         xlim=c(0, max(abs(coeff.full))+0.1),
         xlab=list("Magnitude of effect", cex=1.5), 
         ylab = list("Effect", cex=1.5),
         groups=(coeff.full>0)[coeff$ix], col=c("lightblue", "orange"),
         scales=list(cex=1.5)
)
dev.off()