method <- read.csv('http://datasets.connectmv.com/file/systematic-method.csv')
summary(method)

bitmap('boxplot-for-systematic-method-used-2014.png', pointsize=14, res=300,
       type="png256", width=6, height=5)
par(mar=c(2, 4, 0.2, 0.2))   # (bottom, left, top, right) spacing around plot
boxplot(method, ylab="Grades [maximum available is 28 points]", ylim=c(0, 30))
dev.off()
