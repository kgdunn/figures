A.hist <- read.csv('http://datasets.connectmv.com/file/batch-yields.csv')
A.hist <- A.hist$Yield
N <- length(A.hist)
B <- c(83.5,78.9,82.7,93.2,86.3,74.7,81.6,92.4,83.6,72.4,70.8,77.7,80.7,81.4,86.1,77.9)

#acf(A.hist)

summary(A.hist)
delta <- 16
A.hist.means <- numeric(N-delta+1) 
for (i in 1:(N-delta+1))
{
    A.hist.means[i] <- mean(A.hist[i:(i+delta-1)])
}

A.hist.means.diffs <- numeric(N-delta-delta+1) 
for (i in 1:(N-delta-delta+1))
{
    A.hist.means.diffs[i] <- A.hist.means[i+delta] - A.hist.means[i]   # (mean of group i+1) - (mean of group i)
}

A.last <- A.hist[(N-delta+1):N]
difference <- mean(B) - mean(A.last)
maxrange <- max(ceiling(abs(A.hist.means.diffs)))

plot(A.hist, type="p")

# library(BHH2)
# bitmap('system-comparison-dotplot-grouped-new-extra.png', type="png256", width=12, height=3, res=300, pointsize=14) 
# dotPlot(A.hist.means.diffs, xlab="Difference between means of 2 adjacent groups (16 batches per group)", xlim=c(-maxrange, maxrange))
# lines(x=c(difference, difference), y=c(0,0.35))
# arrows(difference, 0.2, difference+1.6, 0.2, code=2)
# text(difference, 0.40, round(difference,2))
# dev.off()

bitmap('system-comparison-dotplot-grouped-new-extra-time.png', type="png256", width=12, height=7, res=300, pointsize=14) 
par(mar=c(4, 4.2, 2, 0.2))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
plot(A.hist.means.diffs, type="b", ylab="Differences in yield between adjacent groups")
abline(h=difference, col="blue")
abline(h=0, col="grey")
h <- 170
arrows(h, difference, h, difference+1.6, 0.2, code=2, col="blue")
text(h, difference+1.8, "Better performance", col="blue")
dev.off()

sum(A.hist.means.diffs > difference)
length(A.hist.means.diffs)
sum(A.hist.means.diffs > difference) / length(A.hist.means.diffs)


#---------
# Using external variance estimate
#---------------------------
B.mean <- mean(B)
A.mean <- mean(A.last)
difference <- B.mean - A.mean
c_n = qnorm(0.975)
sigma = sd(A.hist)
sigma
na = length(A.last)
nb = length(B)

denom_sigma =  sqrt(sigma^2*(1/na + 1/nb))

z = (difference - 0)/denom_sigma
z
1-pnorm(z)

LB = difference - c_n * sqrt(sigma^2*(1/na + 1/nb))
UB = difference + c_n * sqrt(sigma^2*(1/na + 1/nb))
c(LB, UB)

#---------
# Using intenal variance estimate
#---------------------------
na = length(A.last)
nb = length(B)
B.mean <- mean(B)
A.mean <- mean(A.last)
B.var <- var(B)
A.var <- var(A.last)
dof <- na+nb-2
var.pooled = ((na-1)*A.var + (nb-1)*B.var) / (na+nb-2)

difference <- B.mean - A.mean
c_t = qt(0.975, df = dof)

denom_sigma =  sqrt(var.pooled*(1/na + 1/nb))
denom_sigma
z = (difference - 0)/denom_sigma
z
1-pt(z, df=dof)

# LB = difference - c_t * denom_sigma
# UB = difference + c_t * denom_sigma
# c(LB, UB)
