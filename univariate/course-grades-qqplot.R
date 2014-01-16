grades <- read.csv('~/4N4-grades-2013.csv')

library(car)
bitmap('qqplot-4N4-2013-grades.png', type="png256", width=10, height=10, res=300, pointsize=16)
par(mar=c(2, 4, 1, 0.2))   # (bottom, left, top, right) spacing around plot
qqPlot(grades $G, ylab="Final grades, 4N4, 2013", cex=2)
dev.off()