X <- rbind( c(1,    -1,    -1,    -1),
            c(1,     1,    -1,    -1),
            c(1,    -1,     1,    -1),
            c(1,     1,     1,    -1),
            c(1,    -1,    -1,     1),
            c(1,     1,    -1,     1),
            c(1,    -1,     1,     1),
            c(1,     1,     1,     1) )
X <- cbind(X, X[,2] * X[,3])  # factor D = AB
X <- cbind(X, X[,2] * X[,4])  # factor E = AC
X <- cbind(X, X[,3] * X[,4])  # factor F = BC
X <- cbind(X, X[,2] * X[,3] * X[,4])  # factor G = ABC
y <- as.matrix(c(14.0, 16.8, 15.0, 15.4, 27.6, 24.0, 27.4, 22.6), 8, 1)
t(X) %*% X  # verify that X is orthogonal
b <- solve(t(X) %*% X) %*% t(X) %*% y

# The Pareto plot does not show the intercept term.  Show the bars in sorted order
# of absolute value, which means we must also sort the labels accordingly:
labels <- c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
N = length(b)
b.mod = abs(b[2:N])      # ignore intercept
idx = order(b.mod)       # what is the sorted order?
b.mod = b.mod[idx]
labels.mod = labels[idx] # sort the labels in the same order as b.mod

# Show the Pareto plot
library(lattice)
bitmap('fractional-factorial-question.png', type="png256",
        width=6, height=6, res=300, pointsize=14)
barchart(as.matrix(b.mod), ylab = "Effect", xlab="Magnitude of effect", 
         scales=list(y=list(labels=labels.mod)), col=0)
dev.off()
