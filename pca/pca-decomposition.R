temps<- read.csv('http://openmv.net/file/room-temperature.csv')
X <- data.frame(x1=temps$FrontLeft, x2=temps$FrontRight, x3=temps$BackLeft)
 
library(lattice)
grouper <- c(numeric(length=50)+1, numeric(length=10)+2, numeric(length=144-50-10)+1)
grouper[131] = 3
 
cube <- function(angle){
    # Function to draw the cube at a certain viewing ``angle``
    xlabels = 0
    ylabels = 0
    zlabels = 0
    lattice.options(panel.error=NULL) 
    print(cloud(
                 X$x3 ~ X$x1 * X$x2,
                 cex = 2, 
                 type="p",
                 groups = grouper,
                 pch=20,
                 col=c("black", "blue", "red"),   
                 main="",
                 screen = list(z = angle, x = -70, y = 0),        
                 par.settings = list(axis.line = list(col = "transparent")), 
                 scales = list(
                     col = "black", arrows=TRUE,
                     distance=c(0.5,0.5,0.5)
                 ),
                 xlab="x1",
                 ylab="x2",
                 zlab="x3",
                 zoom = 1.0
             )
        )  
}
 
library(RSvgDevice)
devSVG("pca-decomposition-rotating-cube.svg", width=20, height=10)
cube(12)
dev.off()


# Calculate the mean and standard deviation, ignoring missing values
temps$Date <- c()
temps.mean <- apply(temps, 2, mean, na.rm=TRUE)
temps.sd <- apply(temps, 2, sd, na.rm=TRUE)
temps.mc <- sweep(temps, 2, temps.mean)
temps.mcuv <- sweep(temps.mc, 2, temps.sd, FUN='/')

model.pca <- prcomp(temps, scale=TRUE)
temps.P <- model.pca$rotation
temps.T <- model.pca$x

devSVG("pca-decomposition-score-plot.svg", width=10, height=10)
par(mar=c(5, 4.2, 4.0, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.5, cex.main=1.8, cex.sub=1.5, cex.axis=1.1)
plot(temps.T[,1], temps.T[,2], xlab="Score 1", ylab="Score 2", main="Room temperature data", 
pch=20, cex=3, xlim=c(-3.5,3.5))
lines(temps.T[grouper==2,1], temps.T[grouper==2,2], col="blue", type="p", pch=20, cex=3)
lines(temps.T[grouper==3,1], temps.T[grouper==3,2], col="red", type="p", pch=20, cex=3)
abline(h=0, v=0)
dev.off()

devSVG("pca-decomposition-SPE-plot.svg", width=20, height=7)
A <- 2
order <- seq(1,nrow(temps.T))
temps.E.A <- temps.mcuv - as.matrix(temps.T[,seq(1,A)],N,A) %*% t(temps.P[,seq(1,A)])  # if A=1: 
SPE.A <- apply(temps.E.A ** 2, 1, sum)
reasonable.limit.95 = quantile(c(SPE.A[1:40], SPE.A[60:120]), 0.95)
par(mar=c(3.0, 4.2, 3.0, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.3, cex.main=1.5, cex.sub=1.5, cex.axis=1.5)
plot(SPE.A, type='p', main="SPE after using 2 LV", ylab="SPE", xlab="Time order", pch=20, cex=3)
lines(order[grouper==2], SPE.A[grouper==2], pch=20, type="p", col="blue", cex=2)
lines(order[grouper==3], SPE.A[grouper==3], pch=20, type="p", col="red", cex=2)

abline(h=reasonable.limit.95, col='red')
dev.off()


