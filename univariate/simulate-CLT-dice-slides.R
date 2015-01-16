N = 6000


#Sys.setenv(R_GSCMD='/usr/local/bin/gs-noX11')
bitmap('simulate-CLT-dice-slides.png', type="png256", 
       width=7, height=7, res=300, pointsize=14)

s1 <- as.integer(runif(N, 1, 7))
hist(s1, main="6000 throws from a 6-sided dice", xlab="Value observed", 
     breaks=seq(0,6)+0.5)


dev.off()