d <- c(11, 26, 18, 16, 20, 12,  8, 26, 12, 17, 14)
m <- c(25,  3, 27, 30, 33, 16, 28, 27, 12, 32, 16)

BOD <- c(d, m)
g.d <- matrix(data=0, nrow=length(d), ncol=1)
g.m <- matrix(data=1, nrow=length(m), ncol=1)
g <- rbind(g.d, g.m)
g

model <- lm(BOD ~ g)
summary(model)

# Call:
# lm(formula = BOD ~ g)
# 
# Residuals:
#     Min      1Q  Median      3Q     Max 
# -19.636  -5.114   1.136   5.114  10.364 
# 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   16.364      2.387   6.856 1.16e-06 ***
# g              6.273      3.375   1.858   0.0779 .  
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1 
# 
# Residual standard error: 7.915 on 20 degrees of freedom
# Multiple R-squared: 0.1473,   Adjusted R-squared: 0.1046 
# F-statistic: 3.454 on 1 and 20 DF,  p-value: 0.07788

# Confidence interval for dichotomous "gamma", given by "g"
confint(model)
#                  2.5 %  97.5 %
# (Intercept) 11.3852724 21.3420
# g           -0.7677425 13.3132

# Compare to the t-test:
t.test(m, d, var.equal=TRUE)
# 
#   Two Sample t-test
# 
# data:  m and d 
# t = 1.8585, df = 20, p-value = 0.07788
# alternative hypothesis: true difference in means is not equal to 0 
# 95 percent confidence interval:
#  -0.7677425 13.3131971 
# sample estimates:
# mean of x mean of y 
#  22.63636  16.36364
