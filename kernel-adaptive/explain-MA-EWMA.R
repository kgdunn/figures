stick <- function(x, y, pch=16, linecol=1, clinecol=1,...){
if (missing(y)){
    y = x
    x = 1:length(x) }
    plot(x,y,pch=pch, ...)
    for (i in 1:length(x)){
       lines(c(x[i],x[i]), c(0,y[i]),col=linecol)
    }
    lines(c(x[1]-2,x[length(x)]+2), c(0,0),col=clinecol)
}

N <- 10

MA <- numeric(N)
MA[(N-4):N] = 0.2

EWMA <- numeric(N)
lambda <- 0.3
for (i in 1:N)
{
    EWMA[i] = lambda*(1-lambda)^(N-i)
}


bitmap('explain-MA-EWMA.png', type="png256", width=9, height=5, res=300, pointsize=14)
par(mar=c(2, 2, 1.5, 1))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.2, cex.main=1.5, cex.sub=1.2, cex.axis=1.2)
par(mfrow=c(1,2))

stick(MA, main=(expression(paste("MA weights when w=5"))), ylim=c(0, 1), xaxt="n", xlab="", ylab="", panel.first = grid())
axis(1, at=seq(1, N), labels=c( "", "...", "", "t-7", "", "t-5", "", "t-3","", "t-1"), cex.lab=2)


stick(EWMA, main=(expression(paste("EWMA weights when "*lambda,' = 0.3'))), ylim=c(0, 1), xaxt="n", xlab="", ylab="", panel.first = grid())
axis(1, at=seq(1, N), labels=c("", "...", "", "t-7", "", "t-5", "", "t-3","", "t-1"), cex.lab=2)

dev.off()