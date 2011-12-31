set.seed(13)
rand.f <- rf(1000, df1=200, df=150)  # 1000 values from F-distribution
library(car)

bitmap('qqplot-comparison.png', type="png256", width=14, height=7, res=300, pointsize=14)
m <- t(matrix(seq(1,2), 2, 1))
layout(m)

hist(rand.f, freq=FALSE)                        # looks sort of normally distributed
lines(density(rand.f))

qqPlot(rand.f, distribution="norm", main="q-q plot against the normal distribution")

dev.off()
