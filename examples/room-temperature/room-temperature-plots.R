roomtemp <- read.csv('http://datasets.connectmv.com/file/room-temperature.csv')
summary(roomtemp)
ylim = c(290, 300)

bitmap('room-temperatures.png', pointsize=14, res=300, width=10)
par(mar=c(4, 4, 0.2, 0.2))  # (B, L, T, R) par(mar=c(5, 4, 4, 2) + 0.1)
plot(roomtemp$FrontLeft,   type='l',           col="blue", 
     ylim=ylim, xlab="Sequence order", ylab="Room temperature [K]")
lines(roomtemp$FrontRight, type='b',  pch='o', col="blue")
lines(roomtemp$BackLeft,   type='l',           col="black")
lines(roomtemp$BackRight,  type='b',  pch='o', col="black")

legend(25, 300, legend=c("Front left", "Front right", "Back left", "Back right"),
       col=c("blue", "blue", "black", "black"), lwd=2, pch=c(NA, "o", NA, "o"))
dev.off()