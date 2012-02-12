V <- c(0.01, 0.12, 0.24, 0.38, 0.51, 0.67, 0.84, 1.01, 1.15, 1.31)
T <- c(273, 293, 313, 333, 353, 373, 393, 413, 433, 453)


#bitmap('thermocouple.png', type="png256", width=7, height=7, res=300, pointsize=14) 
#par(mar=c(4, 4.2, 2, 0.2))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
plot(V, T, xlab="Voltage [mV]", ylab="Temperature [K]")
#dev.off()


model <- lm(T ~ V)
summary(model)
coef(model)

plot(V, T)
v.new <- seq(0, 1.5, 0.1)
t.pred <- coef(model)[1] + coef(model)[2]*v.new
lines(v.new, t.pred, type="l", col="blue")