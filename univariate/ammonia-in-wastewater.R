nh4 <- read.csv('http://datasets.connectmv.com/file/ammonia.csv')
summary(nh4$Ammonia)              # just to check we've got the right data
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#    9.99   30.22   36.18   36.09   42.37   58.74

# Investigate the histogram or density plots first
hist(nh4$Ammonia)
plot(density(nh4$Ammonia))

# The qq-plot confirms it is normal, apart from the right-hand-side tail
library(car)
png(file='ammonia-qqplot.png')
qqPlot(nh4$Ammonia)
dev.off()

# Estimate the parameters of the distribution
nh4.mean = mean(nh4$Ammonia)  # 36.09499
nh4.sd = sd(nh4$Ammonia)      # 8.518928

# Advanced: use the MASS package in R to estimate
# (very similar results)
fitdistr(nh4$Ammonia, "normal")


level <- 40

# Using only the data to calculate p(Ammonia > level):
# calculate fraction of samples greater than ``level``
sum(nh4$Ammonia > level) / length(nh4$Ammonia)

# Using the normal distribution to estimate p(Ammonia > level):

# Calculate a z-value first, then the cumulative probability
z <- (level - nh4.mean)/nh4.sd
1 - pnorm(z)

# Or, you can get the answer more directly:
1 - pnorm(level, mean=nh4.mean, sd=nh4.sd)

# More correctly, we should have used the t-distribution,
# because we actually estimated the standard deviation
# We basically get the same answer
1 - pt(z, df=(length(nh4$Ammonia)-1))
