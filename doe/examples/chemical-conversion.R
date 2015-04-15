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

# Conversion: BHH2, p 200
A <- B <- C <- D <- c(-1, +1)
design <- expand.grid(A=A, B=B, C=C, D=D)

A <- design$A
B <- design$B
C <- design$C
D <- design$D

conversion <- c(70, 60, 89, 81, 69, 62, 88, 81, 60, 49, 88, 82, 60, 52, 86, 79)

lm(conversion ~ A*B*C*D)
paretoPlot(lm(conversion ~ A*B*C*D))