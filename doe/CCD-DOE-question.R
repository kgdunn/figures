# | 1         | -1         | -1        |  36        |
# | 2         | +1         | -1        |  45        |
# | 3         | -1         | +1        |  41        |
# | 4         | +1         | +1        |  60        |
# 
# | 7         | 1.41       |  0        |  52        |
# | 8         | 0          | 1.41      |  49        |
# | 9         | -1.41      | 0         |  41        |
# | 10        | 0          | -1.41     |  38        |
# 
# | 0         | 0          | 0         |  53        |
# | 11        | 0          | 0         |  48        |
# | 5         | 0          |  0        |  49        |
# | 6         | 0          |  0        |  51        |

T <- c(-1, +1, -1, +1,  0,  0)
C <- c(-1, -1, +1, +1,  0,  0)
y <- c(36, 45, 41, 60, 53, 49)

mod <- lm(y ~ T + C + T*C)
summary(mod)

T <- c(-1, +1, -1, +1, 1.41,    0, -1.41,     0,  0,  0,  0)
C <- c(-1, -1, +1, +1,    0, 1.41,     0, -1.41,  0,  0,  0)
y <- c(36, 45, 41, 60,   52,   49,    41,    38, 53, 49, 48)

mod <- lm(y ~ T + C + T*C +I(T^2) + I(C^2))
summary(mod)
confint(mod)
# The lower and upper bounds, in coded units, over which we want
# to visualize the surface
bound <- 4.2
N <- 150  # resolution of surface (higher values give smoother plots)
T_plot <- seq(-bound, bound ,length=N)
C_plot <- seq(-bound, bound, length=N)
grd <- expand.grid(T=T_plot, C=C_plot)

# Predict directly from least squares model
grd$y <- predict(mod, grd)

mygray <- function(n) gray(seq(0,n)/18+0.5)
 
library(lattice)
png(file="CCD-DOE-question.png", height = 1500, width = 1500, res=300, bg="transparent")
par.set <-  list(axis.line = list(col = "transparent"), clip = list(panel = "off"))
contourplot(y ~ T * C, 
            data = grd,
            cuts = 10, 
            region = TRUE,
            col.regions = mygray,
            xlab = "Temperature",
            ylab = "Catalyst",
            main = "Contour plot for factors T and C (coded units)"
           )
 
trellis.focus("panel", 1, 1, highlight=FALSE)
lpoints(T, C, pch="O", col="black", cex=1.5)
llines(c(-1, +1, +1, -1, -1), c(-1, -1, +1, +1, -1), col="black", lwd=2)
llines(c(-sqrt(2), sqrt(2)), c(0,0), col="black", lwd=2)
llines(c(0,0), c(-sqrt(2), sqrt(2)), col="black", lwd=2)
trellis.unfocus()
dev.off()