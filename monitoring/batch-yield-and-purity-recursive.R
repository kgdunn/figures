# Thanks to Mudassir for his source code to recursively calculate
# the limits. Some updates were made.
# ----------------------------------------------------------------
data <- read.csv('http://datasets.connectmv.com/file/batch-yield-and-purity.csv')
y <- data$yield
variable <- "Yield"
N <- 3

# No further changes required
# The code below will work for any new data set
subgroups <- matrix(y, N, length(y)/N)
x.mean <- numeric(length(y)/N)
x.sd <- numeric(length(y)/N)

# Calculate mean and sd of subgroups (see R-tutorial)
x.mean <- apply(subgroups, 2, mean)
x.sd <- apply(subgroups, 2, sd)
ylim <- range(x.mean) + c(-5, +5)
k <- 1
doloop <- TRUE

# Prevent infinite loops
while (doloop & k < 5){
  # Original definition for a_n: see course notes
  an <- sqrt(2)*gamma(N/2)/(sqrt(N-1)*gamma((N-1)/2))
  S <- mean(x.sd)
  xdb <- mean(x.mean) # x-double bar
  LCL <- xdb - (3*S/(an*sqrt(N)))
  UCL <- xdb + (3*S/(an*sqrt(N)))
  print(c(LCL, UCL))

  # Create a figure on every loop
  bitmap(paste('batch-yield-phaseI-round-', k, '-', variable, '.png',  sep=""),
          type="png256", width=10, height=4, res=300, pointsize=14)
  par(mar=c(4.2, 4.2, 0.5, 0.5)) 
  par(cex.lab=1.3, cex.main=1.5, cex.sub=1.5, cex.axis=1.5)
  plot(x.mean, type="b", pch=".", cex=5, main="", 
       ylab=paste("Phase I subgroups: round", k), 
       xlab="Sequence order", ylim=ylim)
  abline(h=UCL, col="red")
  abline(h=LCL, col="red")
  abline(h=xdb, col="green")
  lines(x.mean, type="b", pch=".", cex=5)
  dev.off()

  if (!( any(x.mean<LCL) | any(x.mean>UCL) )){
    # Finally!  No more points to exclude
    doloop <- FALSE
  }
  k <- k + 1
  
  # Retain in x.sd and x.mean only those entries 
  # that are within the control limits  
  x.sd <- x.sd[x.mean>=LCL]
  x.mean <- x.mean[x.mean>=LCL]
  x.sd <- x.sd[x.mean<=UCL]
  x.mean <- x.mean[x.mean<=UCL]
} # end: while doloop