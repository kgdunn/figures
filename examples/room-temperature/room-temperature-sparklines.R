roomtemp <- read.csv('http://datasets.connectmv.com/file/room-temperature.csv')

library(YaleToolkit)

bitmap('room-temperature-sparklines.png', res=300, width=5, height=1)
sparklines(roomtemp[,2:5], outer.margin=unit(c(1, 3, 1, 1), "lines"), buffer = unit(0.2, "lines"), xaxis="none", yscale=c(290, 300), ylab=c("FrontLeft"))
dev.off()