# Save stock prices to CSV file:
CNR.TO <- read.csv('stock-prices.csv')
summary(CNR.TO)
library(car)

bitmap('stock-prices-qqplot.png', pointsize=14, res=300, 
       type="png256", width=5, height=5)
par(mar=c(2, 4, 1, 0.2))   # (bottom, left, top, right) spacing around plot
qqPlot(CNR.TO$Adj.Close, ylab="Adjusted closing price for CNR.TO")
dev.off()

# Location and spread
c(mean(CNR.TO$Adj.Close), median(CNR.TO$Adj.Close))
c(sd(CNR.TO$Adj.Close), mad(CNR.TO$Adj.Close))
library(MASS)
fitdistr(CNR.TO$Adj.Close, 'normal')

# Independent? Can we see it visually? Defintely!
bitmap('stock-prices-timeseries.png', pointsize=14, res=300, 
       type="png256", width=10, height=5)
par(mar=c(2, 4, 1, 0.2))   # (bottom, left, top, right) spacing around plot
plot(CNR.TO$Adj.Close, type="l", ylab="Adjusted closing price for CNR.TO")
dev.off()

# Use the xts library for better plots; search the software tutorial for "xts" to see how.
library(xts)
date.order <- as.Date(CNR.TO$Date, format="%Y-%m-%d")
CNR.TO$Date
date.order
Adj.Close <- xts(CNR.TO$Adj.Close, order.by=date.order)
bitmap('stock-prices-timeseries-xts.png', res=300, pointsize=14, width=10, height=5)
par(mar=c(2, 4, 1, 0.2))
plot(Adj.Close, ylab="Adjusted closing price for CNR.TO", main="")
dev.off()


# Use the autocorrelation function (acf) to check lack of independence:
# (we will introduce this function later on)
acf(CNR.TO$Adj.Close, lag=40)
acf(diff(CNR.TO$Adj.Close), lag=5)

# Probability:
1-pnorm(77, mean=mean(CNR.TO$Adj.Close), sd=sd(CNR.TO$Adj.Close))