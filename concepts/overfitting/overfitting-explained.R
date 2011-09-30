x <- c(1.5, 4.5, 5.5, 8, 10)
y <- c(2, 3.5, 7, 8.8, 9.5)

new_x <- data.frame(x = seq(0, 12, 0.1))
no_model <- lm(y~1)
linmod <- lm(y~x)
quad <- lm(y~x+I(x^2))
cubic <- lm(y~x+I(x^2)+I(x^3))
quartic <- lm(y~x+I(x^2)+I(x^3)+I(x^4))
summary(linmod)
summary(quad)
summary(cubic)
summary(quartic)

x_test <- data.frame(x = c(3.0, 5, 7, 9))
set.seed(5)
y_test <- 0.5 + x_test + 0.5*rnorm(4)


e_0 <- y_test - predict(no_model, x_test)
ssq_0 = sum(e_0*e_0)

e_1 <- y_test - predict(linmod, x_test)
ssq_1 = sum(e_1*e_1)

e_2 <- y_test - predict(quad, x_test)
ssq_2 = sum(e_2*e_2)

e_3 <- y_test - predict(cubic, x_test)
ssq_3 = sum(e_3*e_3)

e_4 <- y_test - predict(quartic, x_test)
ssq_4 = sum(e_4*e_4)

c(ssq_0, ssq_1, ssq_2, ssq_3, ssq_4)


bitmap('overfitting-rawdata.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Raw data")
dev.off()

bitmap('overfitting-nomodel.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="No model")
lines(new_x$x, predict(no_model, new_x), type='l', col='black', lw=1)
legend(7, 4, c('No model'), cex=1,
       lwd=c(1), col=c('black'))
dev.off()

bitmap('overfitting-linear.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Linear model")
lines(new_x$x, predict(no_model, new_x), type='l', col='black', lw=1)
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=2)
legend(7, 4, c('No model', 'Linear'), cex=1,
       lwd=c(1, 2), col=c('black', 'blue'))
dev.off()


bitmap('overfitting-quadratic.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Quadratic model")
lines(new_x$x, predict(no_model, new_x), type='l', col='black', lw=1)
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=2)
lines(new_x$x, predict(quad, new_x), type='l', col='purple', lw=3)
legend(7, 4, c('No model', 'Linear', 'Quadratic'), cex=1,
       lwd=c(1,2,3), col=c('black', 'blue', 'purple'))
dev.off()


bitmap('overfitting-cubic.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Cubic model")
lines(new_x$x, predict(no_model, new_x), type='l', col='black', lw=1)
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=2)
lines(new_x$x, predict(quad, new_x), type='l', col='purple', lw=3)
lines(new_x$x, predict(cubic, new_x), type='l', col='orange', lw=4)
legend(7, 4, c('No model', 'Linear', 'Quadratic', 'Cubic'), cex=1,
       lwd=c(1,2,3,4), col=c('black', 'blue', 'purple', 'orange'))
dev.off()


bitmap('overfitting-cubic-testing.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=1.5, lwd=4, main="Testing data shown")
lines(t(x_test), t(y_test), type='p', pch=22, cex=1.5)
lines(new_x$x, predict(cubic, new_x), type='l', col='orange', lw=4)
legend(7, 4, c('Building data', 'Testing data'), cex=0.9, lwd=c(4, 1), col='black')
dev.off()


bitmap('overfitting-quartic.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Quartic model")
lines(new_x$x, predict(no_model, new_x), type='l', col='black', lw=1)
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=2)
lines(new_x$x, predict(quad, new_x), type='l', col='purple', lw=3)
lines(new_x$x, predict(cubic, new_x), type='l', col='orange', lw=4)
lines(new_x$x, predict(quartic, new_x), type='l', col='red', lw=5)
legend(7, 4, c('No model', 'Linear', 'Quadratic', 'Cubic', 'Quartic'), cex=1,
       lwd=c(1,2,3,4,5), col=c('black', 'blue', 'purple', 'orange', 'red'))
dev.off()


# ------------ Another exampl

x <- c(1.5, 4.5, 5.5, 8, 10)
y <- c(2, 3.5, 5, 7.3, 9.5)

new_x <- data.frame(x = seq(0, 12, 0.1))
linmod <- lm(y~x)
quad <- lm(y~x+I(x^2))
cubic <- lm(y~x+I(x^2)+I(x^3))
summary(linmod)
summary(quad)
summary(cubic)
summary(quartic)

bitmap('overfitting-cubic-alternative.png', type="png256", width=5, height=5, res=300, pointsize=14)
par(cex.lab=1.1, cex.main=1.1, cex.sub=1.1, cex.axis=1.1)
par(mar=c(2.0, 2.0, 1.0, 0.5))  # (bottom, left, top, right);
plot(x, y, type='p', xlim=c(1, 11), ylim=c(0, 11), cex=2, lwd=4, main="Cubic model (data set 2)")
lines(new_x$x, predict(linmod, new_x), type='l', col='blue', lw=1)
lines(new_x$x, predict(quad, new_x), type='l', col='purple', lw=2)
lines(new_x$x, predict(cubic, new_x), type='l', col='orange', lw=3)
legend(7, 4, c('Linear', 'Quadratic', 'Cubic'), cex=1,
       lwd=c(2, 3, 4), col=c('blue', 'purple', 'orange'))
dev.off()

