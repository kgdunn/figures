library(RSvgDevice)
devSVG("explain-scatterplot-R.svg", width=10, height=10)

plot(c(2, 5, 6, 9, 5, 3), c(3, 7, 7, 9, 4, 4))
dev.off()