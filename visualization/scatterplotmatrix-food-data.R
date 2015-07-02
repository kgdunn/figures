food <- read.csv('http://openmv.net/file/food-texture.csv')
library(car)

bitmap('scatterplotmatrix-food-data.png', pointsize=14, res=300,  width=7, height=5)

scatterplotMatrix(food[,2:6],      # don't need the non-numeric first column
                  smoother=FALSE)  # hide the smoother and bounds

dev.off()
