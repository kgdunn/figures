# import data
data <- read.csv('http://datasets.connectmv.com/file/batch-yields.csv')

# Determine statistics
summary(data)
yield <- data$Yield
yield.mean <- mean(yield)
yield.sd <- sd(yield)


# Rather use a qqplot with limits
library(car)
png(file='ammonia-qqplot.png')
qqPlot(yield)
dev.off()

# Assuming normal distribution determine probability of x < 60 
z <- (60-yield.mean)/yield.sd 
p <- pnorm(z,0,1)
