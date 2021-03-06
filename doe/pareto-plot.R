library(BHH2)
X <- ffDesMatrix(k=7, gen=list(c(4,1,2), c(5,1,3), c(6,2,3), c(7,1,2,3)))    # 4=12, -5=13 (i.e. 5=-13)
X <- cbind(1, X)
y <- c(77.1, 68.9, 75.5, 72.5, 67.9, 68.5, 71.5, 63.7)
XtX <- t(X) %*% X
Xty <- t(X) %*% y
b = solve(XtX) %*% Xty

b.mod = abs(b[2:8])
idx = order(b.mod)
b.mod = b.mod[idx]
labels=c("A", "B", "C", "D", "E", "F", "G")
labels.mod = labels[idx]
library(lattice)

bitmap('pareto-plot.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)

coeff.full <- b[2:length(b)]
coeff <- sort(abs(coeff.full), index.return=TRUE)
barchart(as.matrix(coeff$x), 
         xlim=c(0, max(abs(coeff.full))+0.1),
         xlab=list("Magnitude of effect", cex=1.5), 
         ylab = list("Effect", cex=1.5),
         groups=(coeff.full>0)[coeff$ix], col=c("lightblue", "orange"),
         scales=list(cex=1.5,y=list(labels=labels[coeff$ix])))


barchart(as.matrix(b.mod), ylab = list("Effect", cex=1.5), 
        xlab=list("Magnitude of effect", cex=1.5), col=0,
        scales=list(cex=1.5,y=list(labels=labels.mod))) 

dev.off()

# ----- Revised approach

A = B = C = c(-1, +1)
design <- expand.grid(A=A, B=B, C=C)
A = design$A
B = design$B
C = design$C
D = A*B
E = A*C
F = B*C
G = A*B*C
y = c(77.1, 68.9, 75.5, 72.5, 67.9, 68.5, 71.5, 63.7)

# Rather use this form:
demo = lm(y ~ A + B + C + D + E + F + G)
summary(demo)

# OK, now we are ready to generate the Pareto plot.
# Let's use a library to do that for us.

# library(pid) <-- best to use this! 
# It is better to uncomment and use the line above.

# But this embedded R script on this website does not have the 
# "pid" library available. So we will load the required function 
# from an external server instead:

source('https://yint.org/paretoPlot.R')

# And now we can generate the plot:
paretoPlot(demo)

# Try getting the results manually:
X_matrix = model.matrix(demo)
XtX <- t(X_matrix) %*% X_matrix
Xty <- t(X_matrix) %*% y
b = solve(XtX) %*% Xty
# -------
bitmap('pareto-plot-pid.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
paretoPlot(demo)
dev.off()

# Half-fraction system
# ----------------------
# library(AlgDesign)
# X <- gen.factorial(2, 4)
# X[,5] = X[,1] * X[,2] * X[,3] * X[,4] 
# colnames(X) <- c("A", "B", "C", "D", "E")

X <- ffDesMatrix(k=5, gen=list(c(5,1,2,3,4)))  # Generator: 5=1234
ff <- ffFullMatrix(X[,1:4], x=c(1,2,3,4), maxInt=4)
y = c(45,71,48,65,68,60,80,65,43,100,45,104,75,86,70,96)
X <- ff$Xa
XtX <- t(X) %*% X
Xty <- t(X) %*% y
b = solve(XtX) %*% Xty

N = length(b)
b.mod = abs(b[2:N])
idx = order(b.mod)
b.mod = b.mod[idx]
labels=colnames(X)[2:N]
labels.mod = labels[idx]
library(lattice)


bitmap('pareto-plot-half-fraction.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)

barchart(as.matrix(b.mod), ylab = list("Effect", cex=1.5), xlab=list("Magnitude of effect", cex=1.5), scales=list(cex=1.5,y=list(labels=labels.mod))) #, col=0)

dev.off()

# Full factorial system
# ----------------------
f <- ffDesMatrix(k=4)
ff <- ffFullMatrix(f, x=c(1,2,3,4), 4)
#X <- ffDesMatrix(k=4, gen=list(c(1,2,3,4)))  # Generator: I=1234    # This creates a wierd XTX matrix: cross pattern
#ff <- ffFullMatrix(X, x=c(1,2,3,4), maxInt=4)
y <- c(45,71,48,65,68,60,80,65,43,100,45,104,75,86,70,96)
X <- ff$Xa
XtX <- t(X) %*% X
Xty <- t(X) %*% y
b = solve(XtX) %*% Xty

A <- design$A
B <- design$B
C <- design$C
D <- design$D
mod.full <- lm( y ~ (A + B + C + D)^4 ) 
coeff.full <- coef(mod.full)[2:length(coef(mod.full))]

# N = length(b)
# b.mod = abs(b[2:N])
# idx = order(b.mod)
# b.mod = b.mod[idx]
# labels=colnames(X)[2:N]
# labels.mod = labels[idx]
# library(lattice)
# 
# labels=c("A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD", "ABCD")
# labels.mod = labels[idx]

bitmap('pareto-plot-full-fraction.png', type="png256", width=7, height=7, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)

library(lattice)
coeff <- sort(abs(coeff.full), index.return=TRUE)
barchart(coeff$x, 
         xlim=c(0, max(abs(coeff.full))+0.1),
         xlab=list("Magnitude of effect", cex=1.5), 
         ylab = list("Effect", cex=1.5),
         groups=(coeff.full>0)[coeff$ix], col=c("lightblue", "orange"),
         scales=list(cex=1.5)
)

#barchart(as.matrix(b.mod), ylab = list("Effect", cex=1.5), 
#        xlab=list("Magnitude of effect", cex=1.5), col=0,
#        scales=list(cex=1.5,y=list(labels=labels.mod))) 
dev.off()