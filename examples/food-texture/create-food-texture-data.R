ldpe <- read.csv('/Users/kevindunn/ConnectMV/Datasets/LDPE/LDPE.csv')
ldpe.sub <- subset(ldpe, select = c('Conv', 'Mn', 'Mw', 'LCB', 'SCB'))
ldpe.pca <- prcomp(ldpe.sub, scale=TRUE)
attach(ldpe.sub)

N = dim(ldpe.sub)[1]

food.texture <- data.frame(Oily=SCB, Density=Mn, Crispy=LCB, Fracture=Conv, Hardness=Mw)
plot(food.texture)

set.seed(63464064)
SCB <- (SCB-26)*10 + 16 + rnorm(N,sd=1) 
Mn  <- (Mn/10 - 2700)*5 + 2700 + rnorm(N,sd=50)
LCB <- round((LCB-0.68)*100)
Conv <- (Conv - 0.12)*3000 + rnorm(N, sd=1)
Mw <- (Mw  -151000) /100 + rnorm(N, sd=5)
food.texture <- data.frame(Oil=round(SCB,1), Density=round(Mn/5)*5, Crispy=LCB, Fracture=60-round(Conv), Hardness=round(Mw))

food.texture <- food.texture[1:50,]

food.texture[24,] = c(17.2,2705,8,60-33,113)

write.csv(food.texture, file = "food-texture.csv", quote=FALSE, row.names = FALSE)
plot(food.texture)
