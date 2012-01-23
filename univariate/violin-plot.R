boards <- read.csv('/Users/kevindunn/ConnectMV/Datasets/Boards/first-100-boards.csv')
 
library(lattice)
bitmap('violin-plot.png', type="pngalpha", res=300, pointsize=14)
bwplot(position ~ thickness, boards,
       panel = function(..., box.ratio) {
           panel.violin(..., col = "transparent",
                        varwidth = FALSE, box.ratio = box.ratio)
           panel.bwplot(..., fill = NULL, box.ratio = .1)
       } )
dev.off()