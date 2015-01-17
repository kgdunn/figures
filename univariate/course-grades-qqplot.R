grades <- read.csv('4N4-2013-final-exam.csv')

library(car)
bitmap('qqplot-4N4-2013-grades.png', type="png256", width=10, height=10, res=300, pointsize=16)
par(mar=c(2, 4, 1, 0.2))   # (bottom, left, top, right) spacing around plot
qqPlot(grades$Grades, ylab="Final grades, 4N4, 2013", cex=2)
dev.off()