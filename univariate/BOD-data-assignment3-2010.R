# Data generated with this part of the code
# -------------------------------------------
set.seed(9256132)
BOD.mean = 14
BOD.sd = 0.4 * BOD.mean
N = 11

dilution <- round(rnorm(N, mean=BOD.mean, sd=BOD.sd))
manometric <- round(rnorm(N, mean=BOD.mean*1.6, sd=1.5*BOD.sd))

library(car)
qqPlot(dilution)
qqPlot(manometric)

# Analysis of the data here:
dilution <-   c(11, 26, 18, 16, 20, 12,  8, 26, 12, 17, 14)
manometric <- c(25,  3, 27, 30, 33, 16, 28, 27, 12, 32, 16)

mean(manometric)
mean(dilution)


bitmap('BOD-comparison-raw-data-alternative.png', type="png256", width=7, height=5, res=250, pointsize=14) 
par(mar=c(4.2, 4.2, 0.2, 0.2))
plot(c(dilution, manometric), ylab="BOD values", xaxt='n')
text(5.5,3, "Dilution")
text(18,3, "Manometric")
abline(v=11.5)
dev.off()


bitmap('BOD-comparison-raw-data.png', type="png256", width=7, height=7, res=250, pointsize=14) 
par(mar=c(4.2, 4.4, 0.2, 0.2))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
plot(dilution, type="p", pch=4, 
    cex=2, cex.lab=1.5, cex.main=1.8, cex.sub=1.8, cex.axis=1.8, 
    ylab="BOD values", xlab="Sample number", xaxt='n',
    ylim=c(0,35), xlim=c(0,11.5), col="darkgreen")
lines(manometric, type="p", pch=16, cex=2, col="blue")
lines(rep(0, N), dilution, type="p", pch=4, cex=2, col="darkgreen")
lines(rep(0, N), manometric, type="p", pch=16, cex=2, col="blue")

abline(v=0.5)

legend(8, 5, pch=c(4, 16), c("Dilution", "Manometric"), col=c("darkgreen", "blue"), pt.cex=2)
dev.off()




bitmap('BOD-comparison-plot.png', type="png256", width=7, height=7, res=250, pointsize=14) 
par(mar=c(4.2, 4.2, 0.2, 0.2))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
plot(dilution-manometric, type="p", ylab="Dilution: Manometric", xlab="Sample number", 
     cex.lab=1.5, cex.main=1.8, cex.sub=1.8, cex.axis=1.8, cex=2)
abline(h=0, col="grey60")
dev.off()

bitmap('BOD-comparison-plot-flipped.png', type="png256", width=7, height=7, res=250, pointsize=14) 
par(mar=c(4.2, 4.2, 0.2, 0.2))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
plot(manometric-dilution, type="p", ylab="Manometric: Dilution", xlab="Sample number", 
     cex.lab=1.5, cex.main=1.8, cex.sub=1.8, cex.axis=1.8, cex=2)
abline(h=0, col="grey60")
dev.off()


# I used the same function in "brittleness-comparison-assignment3-2010.R"
group_difference <- function(groupA, groupB)
{    
    # This function assumes either group has missing data.  Calculate
    # the mean and variance omitting the missing values
    
    A.mean <- mean(groupA[!is.na(groupA)])
    A.var <- var(groupA[!is.na(groupA)])
    A.N <- length(groupA[!is.na(groupA)])

    B.mean <- mean(groupB[!is.na(groupB)])
    B.var <- var(groupB[!is.na(groupB)])
    B.N <- length(groupB[!is.na(groupB)])
    
    print(c(A.mean, B.mean, A.var, B.var, A.N, B.N))
    
    difference <- B.mean - A.mean
    var.DOF <- (A.N - 1 + B.N - 1)
    var.pooled <- ((A.N - 1) * A.var + (B.N - 1) * B.var) / var.DOF
    
    sd.denom <- sqrt(var.pooled *(1/A.N + 1/B.N))
    print(c(difference, var.DOF, var.pooled, sd.denom))
    z <- (difference - 0) / sd.denom
    t.critical <- pt(z, var.DOF)
    
    LB <- difference - qt(0.975, df=var.DOF)*sd.denom
    UB <- difference + qt(0.975, df=var.DOF)*sd.denom
return(list(z, t.critical, LB, UB))
}

group_difference(dilution, manometric)
# z = 1.858495
# t.critical = 0.96106 (1-0.001474538)
# -0.7677425 < mu.diff < 13.31320

t.test(dilution, manometric, alternative="two.sided", mu = 0, paired = FALSE, var.equal = TRUE, conf.level = 0.95)
#   Two Sample t-test
# 
# data:  dilution and manometric 
# t = -1.8585, df = 20, p-value = 0.07788
# alternative hypothesis: true difference in means is not equal to 0 
# 95 percent confidence interval:
#  -13.3131971   0.7677425 
# sample estimates:
# mean of x mean of y 
#  16.36364  22.63636 


# Drop out the second point
dilution[2] <- NA
#manometric[2] <- NA  # no need to remove this point
group_difference(dilution, manometric)
# z = 2.128
# t.critical = 0.9766
# 0.119 < mu.diff < 14.35


t.test(dilution, manometric, alternative="two.sided", mu = 0, paired = FALSE, var.equal = TRUE, conf.level = 0.95)
#   Two Sample t-test
# 
# data:  dilution and manometric 
# t = -3.2374, df = 18, p-value = 0.00457
# alternative hypothesis: true difference in means is not equal to 0 
# 95 percent confidence interval:
#  -15.1703  -3.2297


#========================
# Pairing
#
groupA <- dilution <-  c(11, 26, 18, 16, 20, 12,  8, 26, 12, 17, 14)
groupB <- manometric <- c(25,  3, 27, 30, 33, 16, 28, 27, 12, 32, 16)

diffs <- groupB[!is.na(groupA) & !is.na(groupB)] - groupA[!is.na(groupA) & !is.na(groupB)]

diffs
diffs.mean = mean(diffs)
diffs.sd = sd(diffs)
c(diffs.mean, diffs.sd)
diffs.N = length(diffs)
t.crit = qt(0.975, df=diffs.N-1)
LB = diffs.mean  - t.crit * diffs.sd / sqrt(diffs.N)
UB = diffs.mean  + t.crit * diffs.sd / sqrt(diffs.N)
c(LB, UB)