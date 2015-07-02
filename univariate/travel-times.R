travel <- read.csv('http://openmv.net/file/travel-times.csv')
summary(travel)

# Confirm it is not normally distributed
bitmap('travel-times-totaltime.png', pointsize=14, res=300,
       type="png256", width=10, height=5)
layout(matrix(c(1,2), 1, 2)) # layout plot in a 1x2 matrix
par(mar=c(2, 4, 1, 0.2))   # (bottom, left, top, right) spacing around plot
hist(travel$TotalTime, freq=FALSE, main="Histogram for TotalTime")
lines(density(travel$TotalTime))
library(car)
qqPlot(travel$TotalTime, ylab="Total time (min)")
dev.off()

bitmap('travel-times-maxspeed.png', pointsize=14, res=300,
       type="png256", width=10, height=5)
layout(matrix(c(1,2), 1, 2)) # layout plot in a 1x2 matrix
par(mar=c(2, 4, 1, 0.2))   # (bottom, left, top, right) spacing around plot
hist(travel$MaxSpeed, freq=FALSE, main="Histogram for MaxSpeed")
lines(density(travel$MaxSpeed))
qqPlot(travel$MaxSpeed, ylab="Maximum speed (km/hr)")
dev.off()
