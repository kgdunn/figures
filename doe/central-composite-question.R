y <- c(234, 252, 273, 231, 261, 258, 267, 258,      243,     243,     240,      249)
A <- c( -1,  +1,  -1,  +1,   0,   0,   0,   0,        0, sqrt(2),       0, -sqrt(2))
B <- c( -1,  -1,  +1,  +1,   0,   0,   0,   0, -sqrt(2),       0, sqrt(2),        0)

mod.ccd <- lm ( y ~ A + B + A*B + I(A^2) + I(B^2))
summary(mod.ccd)

N <- 100  # resolution of surface
#  (higher values give smoother plots)

# The lower and upper bounds, in coded units, 
# over which we want
# to visualize the surface
bound <- 3 
A_plot <- seq(-bound, bound, length=N)
B_plot <- seq(-bound, bound, length=N)
grd <- expand.grid(A=A_plot, B=B_plot)

# Predict directly from least squares model
grd$y <- predict(mod.ccd, grd)

library(lattice)
bitmap('central-composite-question-surface.png', type="png256", width=8, 
       height=8, res=300, pointsize=14)
contourplot(y ~ A * B, 
            data = grd,
            cuts = 10, 
            region = TRUE,
            col.regions = terrain.colors,
            xlab = "A",
            ylab = "B",
            main = "Predicted response"
)
trellis.focus("panel", 1, 1, highlight=FALSE)

lpoints(A, B, pch="O", col="red", cex=3)
llines(c(-1, +1, +1, -1, -1), c(-1, -1, +1, +1, -1), 
       col="red", lwd=3)
llines(c(-sqrt(2), +sqrt(2)), c(0, 0), col="red", lwd=3)
llines(c(0, 0), c(-sqrt(2), +sqrt(2)), col="red", lwd=3)
trellis.unfocus()
dev.off()