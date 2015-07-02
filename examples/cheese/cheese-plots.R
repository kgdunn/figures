# Load the data and create training/testing splits
cheese <- read.csv('http://openmv.net/file/cheddar-cheese.csv')
N = dim(cheese)[1]
set.seed(2)
cheese$Random <- rnorm(N, 1)
part1 <- seq(1, dim(cheese)[1], 2)
part2 <- seq(2, dim(cheese)[1], 2)
cols <- c("Acetic", "H2S", "Lactic", "Random")

# Correlation between variables
cor(cheese)

# Create a new cheese with these properties
new.cheese <- data.frame(Acetic=9.0, H2S=4.0, Lactic=1.8, Random=0)
write.csv(cheese, 'cheese-with-random-data.csv')  # to double check calculations in other software packages

# Scatter plot matrix
# -----------------------
library(car)
bitmap('cheese-plots.png', type="png256", width=6, height=6, res=300, pointsize=14)
par(mar=c(1.5, 1.5, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.2, cex.main=1.2, cex.sub=1.2, cex.axis=1.2)
scatterplotMatrix(cheese[,2:6], col=c(1,1,1), smooth=FALSE)
dev.off()

bitmap('cheese-plots-no-random.png', type="png256", width=6, height=6, res=300, pointsize=14)
par(mar=c(1.5, 1.5, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.2, cex.main=1.2, cex.sub=1.2, cex.axis=1.2)
scatterplotMatrix(cheese[,2:5], col=c(1,1,1), smooth=FALSE)
dev.off()

# Multiple linear regression model
# -----------------------
model.lm <- lm(Taste ~ Acetic + H2S + Lactic + Random, data=cheese)
summary(model.lm)
confint(model.lm)
predict(model.lm, new.cheese)  # 29.9
RMSEE.lm <- sqrt(mean((cheese$Taste - predict(model.lm))**2))
# Predictions using the "test set swap" procedure
model.lm1 <- lm(Taste ~ Acetic + H2S + Lactic, data=cheese[part1,])
model.lm2 <- lm(Taste ~ Acetic + H2S + Lactic, data=cheese[part2,])
RMSEP.lm1 <- sqrt(mean((cheese$Taste[part1] - predict(model.lm2, cheese[part1,]))**2))
RMSEP.lm2 <- sqrt(mean((cheese$Taste[part2] - predict(model.lm1, cheese[part2,]))**2))
RMSEP.lm <- mean(c(RMSEP.lm1, RMSEP.lm2))

# Ordinary Least Squares model: H2S selected since it has highest correlation with Taste
# ----------------------------
model.lm.H2S <- lm(Taste ~ H2S, data=cheese)
summary(model.lm.H2S)
confint(model.lm.H2S)
predict(model.lm.H2S, new.cheese) 
RMSEE.lm.H2S <- sqrt(mean((cheese$Taste - predict(model.lm.H2S))**2))
# Predictions
model.H2S.lm1 <- lm(Taste ~ H2S, data=cheese[part1,])
model.H2S.lm2 <- lm(Taste ~ H2S, data=cheese[part2,])
RMSEP.H2S.lm1 <- sqrt(mean((cheese$Taste[part1] - predict(model.H2S.lm2, cheese[part1,]))**2))
RMSEP.H2S.lm2 <- sqrt(mean((cheese$Taste[part2] - predict(model.H2S.lm1, cheese[part2,]))**2))
RMSEP.H2S.lm <- mean(c(RMSEP.H2S.lm1, RMSEP.H2S.lm2))

#Neural network model
#--------------------
library(neuralnet)
model.nn <- neuralnet(Taste ~ Acetic + H2S + Lactic + Random, cheese, 
                      hidden=1, 
                      likelihood=TRUE,
                      algorithm="rprop+",
                      linear.output = TRUE)  # because we want predictions of y, not a 0/1 classifier
yhat <- compute(model.nn, cheese[,c(2,3,4,6)])$net.result 
yhat <- model.nn$net.result[[1]][,1]  # two ways of getting the neural network predictions
resid <- cheese$Taste - yhat
RMSEE.nn <- sqrt(mean(resid**2))
R2.nn <- 1 - var(resid) / var(cheese$Taste )
compute(model.nn, new.cheese)   # 38.8

# Predictions, using test set swap
model.nn1 <- neuralnet(Taste ~ Acetic + H2S + Lactic + Random, cheese[part1,], hidden=1, linear.output = TRUE) 
# model.nn2 <- neuralnet(Taste ~ Acetic + H2S + Lactic + Random, cheese[part2,], hidden=1, linear.output = TRUE, algorithm="rprop+", rep=3)   
# Previous model doesn't converge with that particular training set. Add one extra observation to the part 2 data: converges now
model.nn2 <- neuralnet(Taste ~ Acetic + H2S + Lactic + Random, cheese[c(1, part2),], hidden=1, linear.output = TRUE)   
yhat.nn.2 <- compute(model.nn1, cheese[part2, c(2,3,4,6)])$net.result   # predict part2's data, given model built from part 1
yhat.nn.1 <- compute(model.nn2, cheese[part1, c(2,3,4,6)])$net.result   # predict part1's data, given model built from part 2
RMSEP.nn.2 <- sqrt(mean((cheese$Taste[part2] - yhat.nn.2)**2))  # 14.9
RMSEP.nn.1 <- sqrt(mean((cheese$Taste[part1] - yhat.nn.1)**2))
RMSEP.nn <- mean(c(RMSEP.nn.1, RMSEP.nn.2))

# PCR model (one component): long way (manually create PCR model, then linear regression model)
# -----------------------
model.pca <- prcomp(cheese[,cols], scale=TRUE)
summary(model.pca)
P <- model.pca$rotation
T <- model.pca$x
model.pcr <- lm(cheese$Taste ~ T[,1])
summary(model.pcr)
RMSEE.pcr <- sqrt(mean((cheese$Taste - predict(model.pcr))**2))

# regression coefficients
P[,1] * model.pcr$coef[2] / model.pca$scale

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

# Plot the PCA score and the taste value in the scatter plot matrix
T.1 <- model.pca.1$x
# Create a new cheese with these properties
t1 <- T[,1]
new.cheese <- cbind(cheese, t1)
bitmap('cheese-plots-with-t1.png', type="png256", width=6, height=6, res=300, pointsize=14)
par(mar=c(1.5, 1.5, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.2, cex.main=1.2, cex.sub=1.2, cex.axis=1.2)
scatterplotMatrix(new.cheese[,c(2:5,7)], col=c(1,1,1), smooth=FALSE)
dev.off() 

# Scatter plot matrix
# -----------------------
library(car)
bitmap('cheese-plots.png', type="png256", width=6, height=6, res=300, pointsize=14)
par(mar=c(1.5, 1.5, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.2, cex.main=1.2, cex.sub=1.2, cex.axis=1.2)
scatterplotMatrix(cheese[,2:6], col=c(1,1,1), smooth=FALSE)
dev.off()


# PCR using the "pls" package (gets the same results as the longer code above)
# ----------------------------
library(pls)
model.pcr.quick <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="svdpc", scale=TRUE)
summary(model.pcr.quick)
# R2_taste, using 1PC = 62.04%
RMSEE.pcr <- sqrt(mean((cheese$Taste - predict(model.pcr.quick)[,,1])**2))
predict.mvr(model.pcr.quick, new.cheese)  # 54.3


model.pcr.quick.1 <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="svdpc", scale=TRUE, subset=part1)
model.pcr.quick.2 <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="svdpc", scale=TRUE, subset=part2)

RMSEP.pcr.quick1 <- sqrt(mean((cheese$Taste[part1] - predict(model.pcr.quick.2, cheese[part1,])[,,1])**2))
RMSEP.pcr.quick2 <- sqrt(mean((cheese$Taste[part2] - predict(model.pcr.quick.1, cheese[part2,])[,,1])**2))
RMSEP.pcr <- mean(c(RMSEP.pcr.quick1, RMSEP.pcr.quick2))

# Regression coefficients with 1 PC
coef(model.pcr.quick, ncomp=1)

# PLS using the "pls" package
# ----------------------------
library(pls)
model.pls <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="simpls", scale=TRUE)
summary(model.pls)
# R2_taste, using 1PC = 62.7%
RMSEE.pls <- sqrt(mean((cheese$Taste - predict(model.pls)[,,1])**2))
predict.mvr(model.pls, new.cheese) 

model.pls.1 <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="simpls", scale=TRUE, subset=part1)
model.pls.2 <- mvr(Taste ~ Acetic + H2S + Lactic + Random, data=cheese, method="simpls", scale=TRUE, subset=part2)

RMSEP.pls.1 <- sqrt(mean((cheese$Taste[part1] - predict(model.pls.2, cheese[part1,])[,,1])**2))
RMSEP.pls.2 <- sqrt(mean((cheese$Taste[part2] - predict(model.pls.1, cheese[part2,])[,,1])**2))
RMSEP.pls <- mean(c(RMSEP.pls.1, RMSEP.pls.2))

# Coefficients and reliability intervals from ProMV 11.08,  but divide them by "model.pls$scale" and times by sd(y)
#               Coeff               Lower bound         Upper bound
# H2S       1   0.341649600476995   0.214433534821241   0.46886566613275
# Lactic    2   0.318015562701819   0.237906914060081   0.398124211343557
# Acetic    3   0.248419848376738   0.176929971367371   0.319909725386105
# Random	4	0.126124729088615	-0.034083388621767	0.286332846798997

c(0.3416, 0.3180, 0.2484, 0.1261) / model.pls$scale * sd(cheese$Taste)