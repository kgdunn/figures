data <- read.csv('http://openmv.net/file/blender-efficiency.csv')
summary(data)
cov(data)
model<-lm(data$BlendingEfficiency ~ data$ParticleSize + data$MixerDiameter + 
                                     data$MixerRotation + data$BlendingTime)
summary(model)

confint(model, level=0.90)

library(car)
qqPlot(model, id.n=1)

# Omit point 8:
model.update <- lm(model, subset=-c(8))
summary(model.update)
qqPlot(model.update)
confint(model.update, level=0.90)