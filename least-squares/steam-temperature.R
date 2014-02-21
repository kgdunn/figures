
Temp <- c(267, 268, 272, 273, 278, 281, 283, 288, 289, 293, 296)
Steam <- c(220, 251, 211, 210, 155, 152, 122, 157, 100, 64, 58)

plot(Temp, Steam)

model.ls <- lm(Steam ~ Temp)
summary(model.ls)
confint(model.ls)