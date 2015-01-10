library(car)
data(Vocab)


Sys.setenv(R_GSCMD ='/usr/local/bin/gs')
bitmap(file='scatterplots-with-jitter.png', type="png256", res=300, pointsize=14, width=12, height=6)

m <- matrix(1:2, 1, 2)  # Plot layout
layout(m)

# Plot of temperature vs vapour pressure
plot(Vocab$education, Vocab$vocabulary, xlab="Education [years]", 
     ylab="Vocabulary", type="p", cex=0.5)

# Plot of white hairs vs BMD
plot(jitter(Vocab$education), jitter(Vocab$vocabulary), 
     xlab="Education [years]", 
     ylab="Vocabulary",
     cex=0.5)
dev.off()
