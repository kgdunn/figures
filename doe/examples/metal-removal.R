paretoPlot <- function(lsmodel){
  # This code draws a Pareto plot; it requires the "ggplot2" library
  library(ggplot2)
  
  # Extract all the coefficients, except for the intercept
  coeff.full <- coef(lsmodel)[2:length(coef(lsmodel))]
  
  # Return the absolute values of the coefficients
  coeff.abs <- unname(abs(coeff.full))
  
  coeff <- sort(coeff.abs, index.return=TRUE)
  grouping <- unname((coeff.full>0)[coeff$ix])
  grouping[grouping==FALSE]="Negative"
  grouping[grouping==TRUE]="Positive" 
  temp <- names(coeff.full)[coeff$ix]
  fnames <- factor(temp, levels=temp, ordered=TRUE)
  group.colors <- c("Negative" = "grey", "Positive" = "black")
  
  dat <- data.frame(
    label=fnames,
    value=coeff$x,
    group=grouping
  )
  p <- ggplot(dat, aes(x=label, y=value, fill=group)) + 
    geom_bar(stat="identity") +
    coord_flip() + theme_bw() +
    scale_fill_manual(values=group.colors,name = "Sign of coefficient") +
    xlab("Effect") +
    ylab("Magnitude of effect") + 
    ggtitle("Pareto plot")
  p          # Execute the plot (i.e. draw it!)
  return(p)  # Return the plot, so user can continue to modify it
}

A <- B <- C <- D <- c(-1, +1)
design <- expand.grid(A=A, B=B, C=C, D=D)

A <- design$A
B <- design$B
C <- design$C
D <- design$D

metal <- c(0.16, 0.78, 0.18, 1.04, 0.68, 0.79, 0.90, 1.19, 0.19, 0.31, 0.25, 1.35, 0.55, 0.71, 1.19, 1.55)
metal <- metal*100
lm(metal ~ A*B*C*D)
paretoPlot(lm(metal ~ A*B*C*D))

# Select only the observations with C at the - level; I want 8 expts in factors A, B and D 
subset <- c(1, 2, 3, 4, 9, 10, 11, 12)
subset <- c(5, 6, 7, 8, 13, 14, 15, 16)
subset <- c(1, 2, 3, 4, 5, 6, 7, 8)
y.sub <- metal[-subset]
As <- A[-subset]
Bs <- B[-subset]
Cs <- C[-subset]  # note that factor C was the former factor D
lm(y.sub ~ As*Bs*Cs)

# OK, now I have 8 experiments; now pretend I only did a half-fraction
subset <- c(2, 3, 5, 8)
Ah <- As[-subset]
Bh <- Bs[-subset]
Ch <- Cs[-subset]
y.sub.sub <- y.sub[-subset]

lm(y.sub.sub ~ Ah*Bh*Ch)
lm(y.sub ~ As*Bs*Cs)

