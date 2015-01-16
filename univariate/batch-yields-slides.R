# import data
data <- read.csv('batch-yields-slides.csv')

# Determine statistics
summary(data)
yield <- data$Yield
yield.mean <- mean(yield)
yield.sd <- sd(yield)

bitmap('batch-yields-slides.png', type="png256", 
       width=7, height=7, res=300, pointsize=14)
hist(yield, breaks="Scott", xlab="Yield from the batch reactor [g/L]", main="")
dev.off() 

bitmap('batch-yields-good-slides.png', type="png256", 
       width=7, height=7, res=300, pointsize=14)
yield <- (yield-160)/2+160
yield[yield<148]<- NA
hist(yield, breaks="Scott", xlab="Yield from the batch reactor [g/L]", main="")
dev.off() 

bad_yields <- (5-rf(1825,100,100)*2)*30+70
bitmap('batch-yields-poor-slides.png', type="png256", 
       width=7, height=7, res=300, pointsize=14)
hist(bad_yields, breaks="Scott", xlab="Yield from another batch reactor [g/L]",
     main ="")
dev.off()