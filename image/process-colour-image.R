library(png)
X <- readPNG('/Users/kevindunn/ConnectMV/Figures/image/RGB_24bits_palette_sample_image.png', native=FALSE)
X <- round(X * 255)

subsamp = seq(1, 29976, 30)


r <- as.vector(X[,,1])
g <- as.vector(X[,,2])
b <- as.vector(X[,,3])

X <- data.frame(R=r[subsamp], G=g[subsamp], B=b[subsamp])
library(car)
library(car)
bitmap('image-scatterplotmatrix.png', type="png256", width=6, height=6, res=300, pointsize=14)
par(mar=c(1.5, 1.5, 1.5, 0.5))  # (bottom, left, top, right); defaults are par(mar=c(5, 4, 4, 2) + 0.1)
par(cex.lab=1.2, cex.main=1.2, cex.sub=1.2, cex.axis=1.2)
spm(X, smooth=FALSE)
dev.off()

