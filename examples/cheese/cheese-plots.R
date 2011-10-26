# Load the data and create training/testing splits
#cheese <- read.csv('http://datasets.connectmv.com/file/cheddar-cheese.csv')
cheese <- read.csv('C:/kgd61600/My-projects/Datasets/cheese/cheddar-cheese.csv')
N = dim(cheese)[1]
set.seed(0)
cheese$Random <- rnorm(N, 1)
part1 <- seq(1, dim(cheese)[1], 2)
part2 <- seq(2, dim(cheese)[1], 2)
cols <- c("Acetic", "H2S", "Lactic", "Random")

# PCR 
# LS
# PLS (and convert model to coefficients)
# NN
# Do a testing/training switch and estimate RMSEP

#Add column of random numbers still

library(car)

# Scatter plot matrix
# -----------------------
#bitmap('cheese-plots.png', type="png256", width=6, height=6, res=300, pointsize=14)
#par(mar=c(1.5, 1.5, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
#par(cex.lab=1.5, cex.main=1.5, cex.sub=1.5, cex.axis=1.5)
#scatterplotMatrix(cheese[,2:6], col=c(1,1,1), smooth=FALSE)
#dev.off()

# Least squares model
# -----------------------
model.lm <- lm(Taste ~ Acetic + H2S + Lactic + Random, data=cheese)
summary(model.lm)
RMSEE.lm <- sqrt(mean((cheese$Taste - predict(model.lm))**2))
# Predictions
model.lm1 <- lm(Taste ~ Acetic + H2S + Lactic, data=cheese[part1,])
model.lm2 <- lm(Taste ~ Acetic + H2S + Lactic, data=cheese[part2,])
RMSEP.lm1 <- sqrt(mean((cheese$Taste[part1] - predict(model.lm2, cheese[part1,]))**2))
RMSEP.lm2 <- sqrt(mean((cheese$Taste[part2] - predict(model.lm1, cheese[part2,]))**2))
RMSEP.lm <- mean(c(RMSEP.lm1, RMSEP.lm2))

# Least squares model: H2S
# -----------------------
model.lm.H2S <- lm(Taste ~ H2S, data=cheese)
summary(model.lm.H2S)
RMSEE.lm.H2S <- sqrt(mean((cheese$Taste - predict(model.lm.H2S))**2))
# Predictions
model.H2S.lm1 <- lm(Taste ~ H2S, data=cheese[part1,])
model.H2S.lm2 <- lm(Taste ~ H2S, data=cheese[part2,])
RMSEP.H2S.lm1 <- sqrt(mean((cheese$Taste[part1] - predict(model.H2S.lm2, cheese[part1,]))**2))
RMSEP.H2S.lm2 <- sqrt(mean((cheese$Taste[part2] - predict(model.H2S.lm1, cheese[part2,]))**2))
RMSEP.H2S.lm <- mean(c(RMSEP.H2S.lm1, RMSEP.H2S.lm2))


# Least squares model: lactic
# -----------------------
model.lm.lactic <- lm(cheese$Taste ~  cheese$Lactic )
summary(model.lm.lactic)
RMSEE.lm.lactic <- sqrt(mean((cheese$Taste - predict(model.lm.lactic))**2))

# Least squares model: acetic
# -----------------------
model.lm.acetic <- lm(cheese$Taste ~ cheese$Acetic )
summary(model.lm.acetic)
RMSEE.lm.acetic <- sqrt(mean((cheese$Taste - predict(model.lm.acetic))**2))

print(c(RMSEE.lm, RMSEE.lm.acetic, RMSEE.lm.H2S, RMSEE.lm.lactic))

#Neural network model
#--------------------
# model.nn <- nnet(cheese[,2:4], cheese[,5], size = 5)

#library(neuralnet)
#model.nn <- neuralnet(Taste ~ Acetic + H2S + Lactic + Random, cheese, hidden=2, linear.output = FALSE, algorithm="backprop", act.fct="logistic")
#yhat <- model.nn$net.result

# PCR model (one component)
# -----------------------
model.pca <- prcomp(cheese[,cols], scale=TRUE)
summary(model.pca)
P <- model.pca$rotation
T <- model.pca$x
model.pcr <- lm(cheese$Taste ~ T[,1])
summary(model.pcr)
RMSEE.pcr <- sqrt(mean((cheese$Taste - predict(model.pcr))**2))

# Model for part 1
model.pca.1 <- prcomp(cheese[part1,cols], scale=TRUE)
P.1 <- model.pca.1$rotation
T.1 <- model.pca.1$x
model.pcr.1 <- lm(cheese$Taste[part1] ~ T.1[,1])
# Model for part 2
model.pca.2 <- prcomp(cheese[part2,cols], scale=TRUE)
P.2 <- model.pca.2$rotation
T.2 <- model.pca.2$x
model.pcr.2 <- lm(cheese$Taste[part2] ~ T.2[,1])

# Center and scale the new data, using the training set's centering and scaling vectors
x.2.new <- scale(cheese[part2,cols], center= model.pca.1$center, scale=model.pca.1$scale)
x.1.new <- scale(cheese[part1,cols], center= model.pca.2$center, scale=model.pca.2$scale)
# Project the testing data (.new matrices) onto the opposite model
T.2.hat <- x.2.new %*% P.1
T.1.hat <- x.1.new %*% P.2

predict(model.pcr.1)
# Predictions
pred.2 <- model.pcr.1$coef[1] + model.pcr.1$coef[2] * T.2.hat[,1]
pred.1 <- model.pcr.2$coef[1] + model.pcr.2$coef[2] * T.1.hat[,1]
RMSEP.pcr.lm1 <- sqrt(mean((cheese$Taste[part1] - pred.1)**2))
RMSEP.pcr.lm2 <- sqrt(mean((cheese$Taste[part2] - pred.2)**2))
RMSEP.pcr <- mean(c(RMSEP.pcr.lm1, RMSEP.pcr.lm2))

# PCR using the "pls" package (gets the same results as the longer code above)
# ----------------------------
library(pls)
model.pcr.quick <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="svdpc", scale=TRUE)
summary(model.pcr.quick)
# R2_taste, using 1PC = 58.01%
RMSEE.pcr <- sqrt(mean((cheese$Taste - predict(model.pcr.quick)[,,1])**2))

model.pcr.quick.1 <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="svdpc", scale=TRUE, subset=part1)
model.pcr.quick.2 <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="svdpc", scale=TRUE, subset=part2)

RMSEP.pcr.quick1 <- sqrt(mean((cheese$Taste[part1] - predict(model.pcr.quick.2, cheese[part1,])[,,1])**2))
RMSEP.pcr.quick2 <- sqrt(mean((cheese$Taste[part2] - predict(model.pcr.quick.1, cheese[part2,])[,,1])**2))
RMSEP.pcr <- mean(c(RMSEP.pcr.quick1, RMSEP.pcr.quick2))

# PLS using the "pls" package
# ----------------------------
library(pls)
model.pls <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="simpls", scale=TRUE)
summary(model.pls)
# R2_taste, using 1PC = 62.7%
RMSEE.pls <- sqrt(mean((cheese$Taste - predict(model.pls)[,,1])**2))

model.pls.1 <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="simpls", scale=TRUE, subset=part1)
model.pls.2 <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="simpls", scale=TRUE, subset=part2)

RMSEP.pls.1 <- sqrt(mean((cheese$Taste[part1] - predict(model.pls.2, cheese[part1,])[,,1])**2))
RMSEP.pls.2 <- sqrt(mean((cheese$Taste[part2] - predict(model.pls.1, cheese[part2,])[,,1])**2))
RMSEP.pls <- mean(c(RMSEP.pls.1, RMSEP.pls.2))


# Plot observed against predicted
# -----------------------
#plot(cheese$Taste, predict(model.pcr.2))
#abline(lm(predict(model.pcr.2) ~ cheese$Taste -1))
