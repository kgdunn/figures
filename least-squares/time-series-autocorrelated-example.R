e<-read.csv('http://openmv.net/file/gas-furnace.csv')

bitmap('time-series-cycle.png', type="png256", width=14, height=4, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
#layout(matrix(c(1,2), 1, 2))
plot(e$InputGasRate+10, type="l", cex.lab=1.5, cex.main=1.2, cex.sub=1.8, cex.axis=1.8, 
     main="Gas flow rate over time",
     xlab="Time",
     ylab="Gas flow rate")

dev.off()



