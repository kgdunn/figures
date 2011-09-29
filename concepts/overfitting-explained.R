x <- c(1.5, 4.5, 5.5, 8, 10)
y <- c(2, 3.5, 7, 8.8, 9.5)

new_x <- data.frame(x = seq(0, 12, 0.1))
linmod <- lm(y~x)
quad <- lm(y~x+I(x^2))
cubic <- lm(y~x+I(x^2)+I(x^3))
quartic <- lm(y~x+I(x^2)+I(x^3)+I(x^4))
summary(linmod)
summary(quad)
summary(cubic)
summary(quartic)


bitmap('overfitting-rawdata.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Raw data")
dev.off()


bitmap('overfitting-linear.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Linear model")
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=1)
legend(7, 4, c('Linear'), cex=1.3,
       lwd=c(1), col=c('blue'))
dev.off()


bitmap('overfitting-quadratic.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Quadratic model added")
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=1)
lines(new_x$x, predict(quad, new_x), type='l', col='purple', lw=2)
legend(7, 4, c('Linear', 'Quadratic'), cex=1,
       lwd=c(1,2), col=c('blue', 'purple'))
dev.off()


bitmap('overfitting-cubic.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Cubic model added")
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=1)
lines(new_x$x, predict(quad, new_x), type='l', col='purple', lw=2)
lines(new_x$x, predict(cubic, new_x), type='l', col='orange', lw=3)
legend(7, 4, c('Linear', 'Quadratic', 'Cubic'), cex=1,
       lwd=c(1,2,3), col=c('blue', 'purple', 'orange'))
dev.off()


bitmap('overfitting-quartic.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Quartic model added")
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=1)
lines(new_x$x, predict(quad, new_x), type='l', col='purple', lw=2)
lines(new_x$x, predict(cubic, new_x), type='l', col='orange', lw=3)
lines(new_x$x, predict(quartic, new_x), type='l', col='red', lw=4)
legend(7, 4, c('Linear', 'Quadratic', 'Cubic', 'Quartic'), cex=1,
       lwd=c(1,2,3,4), col=c('blue', 'purple', 'orange', 'red'))
dev.off()