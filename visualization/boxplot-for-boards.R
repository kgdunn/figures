boards <- read.csv('http://openmv.net/file/six-point-board-thickness.csv')
summary(boards)

plot(boards[1:100,5], type='l')
plot(boards[1:100,5], type='l')
first100 <- boards[1:100, 2:7]

# Ignore the first date/time column: using only Pos1, Pos2, ... Pos6 columns

bitmap('boxplot-for-two-by-six-100-boards.png', pointsize=14, res=300, 
       type="png256", width=6, heigh=5)
par(mar=c(2, 4, 0.2, 0.2))  # (bottom, left, top, right) spacing around plot
boxplot(first100, ylab="Thickness [mils]")
dev.off()
