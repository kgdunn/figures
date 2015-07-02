gas <- read.csv('http://openmv.net/file/gas-furnace.csv')
summary(gas)

library(car)
bitmap('CO2-gas-furnace-raw-data.png', type="png256", 
        width=6, height=6, res=300, pointsize=14)

# Use the "sp" (scatterplot) function from the "car" library
sp(gas$InputGasRate, gas$CO2, xlab="Gas flow rate", ylab="CO2",
   main="Scatterplot with smoother, spread, and L/S line")
dev.off()

# (Co)variance and correlation
cov(gas)
cor(gas)

# Linear model:
model <- lm(gas$CO2 ~ gas$InputGasRate)
summary(model) 

# ANOVA values
y.mean <- mean(gas$CO2)
RegSS <- sum((predict(model) - y.mean)^2)
RSS <- sum(residuals(model)^2)
TSS <- sum((gas$CO2 - y.mean)^2)
mean.square.residual <- RSS / model$df.residual

# Test normality of residuals
bitmap('CO2-gas-furnace-residuals.png', type="png256", 
        width=6, height=6, res=300, pointsize=14)
par(mar=c(4.2, 4.2, 0.5, 0.5)) 
qqPlot(model)  # the qqPlot "knows" what to do with a model object
dev.off()
