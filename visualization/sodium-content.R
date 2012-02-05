sodium <- c(330, 290, 270, 260, 240, 240, 210, 210, 200, 200, 160, 135, 0)
mean(sodium)
median(sodium)
sd(sodium)
mad(sodium)
summary(sodium)

bitmap('sodium-content.png', pointsize=14, res=300, 
       type="png256", width=6, heigh=5)
par(mar=c(2, 4, 0.2, 0.2))  # (bottom, left, top, right) spacing around plot
boxplot(sodium, ylab="Sodium content [mg] per 50g serving")
dev.off()