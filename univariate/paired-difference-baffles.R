A <- c(77, 74, 83, 74, 81)
B <- c(76, 78, 81, 73, 73)


n <- length(A)
DOF <- n-1
ct <- qt(0.975, DOF)
w <- B-A
wbar = mean(w)
LB = wbar - ct*sd(w)/sqrt(n)
UB = wbar + ct*sd(w)/sqrt(n)
t.test(A, B, paired=TRUE)
