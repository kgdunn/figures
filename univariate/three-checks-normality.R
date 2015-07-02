library(car)

# 1. 
grades <- read.csv('4N4-2013-final-exam.csv')

bitmap('Grades-TS.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
plot(grades$Grades, xlab="", ylab="Grades from a course", cex.lab=2)
dev.off()

bitmap('Grades-hist.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
hist(grades$Grades, xlab="Grades from a course", main="", cex.lab=2)
dev.off()

bitmap('Grades-qqplot.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
qqPlot(grades$Grades, ylab="Grades from a course")
dev.off()

# 2.
furnace <- read.csv('http://openmv.net/file/gas-furnace.csv')
summary(furnace)

bitmap('Furnace-TS.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
plot(furnace$CO2, xlab="", ylab="CO2 from a furnace", cex.lab=2)
dev.off()

bitmap('Furnace-hist.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
hist(furnace$CO2, breaks="Freedman-Diaconis", xlab="CO2 from a furnace", main="", cex.lab=2)
dev.off()

bitmap('Furnace-qqplot.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
qqPlot(furnace$CO2, ylab="CO2 from a furnace")
dev.off()

#3.
set.seed(10)
rand.f <- rf(250, df1=200, df=150) # 80 values from F-distribution hist(rand.f)
bitmap('Rand-TS.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
plot(rand.f, ylab="Unknown data source", cex.lab=2, xlab="")
dev.off()

bitmap('Rand-hist.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
hist(rand.f,breaks=8, xlab="Unknown data source", main="", cex.lab=1.5)
dev.off()

bitmap('Rand-qqplot.png', type="png256", width=7, height=7, res=300, pointsize=16)
par(mar=c(4, 4.5, 0.2, 0.2)) 
qqPlot(rand.f, ylab="Unknown data source")
dev.off()


# library(car)
# bitmap('qqplot-4N4-2013-grades.png', type="png256", width=10, height=10, res=300, pointsize=16)
# par(mar=c(2, 4, 1, 0.2))   # (bottom, left, top, right) spacing around plot
# qqPlot(grades$Grades, ylab="Final grades, 4N4, 2013", cex=2)
# dev.off()