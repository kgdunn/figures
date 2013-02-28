cam <- read.csv('least-squares/cameras.csv')
plot(cam$Cameras, cam$Deaths)
grid()
#identify(cam$Cameras, cam$Deaths, cam$Country)

model.all <- lm(cam$Deaths ~ cam$Cameras)
summary(model.all)
confint(model.all)

# Non-parametric model
plot(cam$Cameras, cam$Deaths)
grid()
lines(lowess(cam$Cameras, cam$Deaths))

library(car)
qqPlot(resid(model.all), labels=cam$Country, id.method="identify")

influenceIndexPlot(model.all)