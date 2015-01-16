N = 40680
set.seed(7)
data <- cut (runif(N), breaks=c(15160, 40680))
data[0]

bitmap('histogram-housework-hamilton.png', type="png256", 
       width=7, height=7, res=300, pointsize=14)
par(mar=c(5, 4.2, 4, 2) + 0.1)  # (B, L, T, R); 
barplot(c(15160, 40680), names.arg=c("Male [15,160]", "Female [40,680]"), 
        ylim=c(0, 50000),
        main="Unpaid housework, Hamilton",
     cex.axis=1.5, cex.lab=1.5, cex.names=1.5, col="white")
dev.off()

bitmap('histogram-housework-hamilton-relative.png', type="png256", 
       width=7, height=7, res=300, pointsize=14)
par(mar=c(5, 4.2, 4, 2) + 0.1)  # (B, L, T, R); 
barplot(c(15160, 40680)/(15160+40680), 
        names.arg=c("Male [15,160]", "Female [40,680]"), 
        main="Unpaid housework, Hamilton", ylim=c(0, 0.8), ylab="Relative frequency",
        cex.axis=1.5, cex.lab=1.5, cex.names=1.5, col="white")
dev.off()


bitmap('histogram-housework-toronto.png', type="png256", 
       width=7, height=7, res=300, pointsize=14)
par(mar=c(5, 4.2, 4, 2) + 0.1)  # (B, L, T, R); 
barplot(c(90865, 266600), names.arg=c("Male [90,865]", "Female [266,600]"), 
        ylim=c(0, 270000),
        main="Unpaid housework, Toronto",
        cex.axis=1.5, cex.lab=1.5, cex.names=1.5, col="white")
dev.off()

bitmap('histogram-housework-toronto-relative.png', type="png256", 
       width=7, height=7, res=300, pointsize=14)
par(mar=c(5, 4.2, 4, 2) + 0.1)  # (B, L, T, R); 
barplot(c(90865, 266600)/357465, names.arg=c("Male [90,865]", "Female [266,600]"), 
        main="Unpaid housework, Toronto", ylim=c(0, 0.8), ylab="Relative frequency",
        cex.axis=1.5, cex.lab=1.5, cex.names=1.5, col="white")
dev.off()



