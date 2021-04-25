A = B  = c(-1, +1)
design <- expand.grid(A=A, B=B)
Butter = design$A
Flour = design$B
y = c(15, 16, 20, 25)
lsmodel = lm(y~Butter*Flour)
summary(lsmodel)

xlim=c(-1.5, +1.5)
ylim=c(-1.5, +1.5)
xlab="Butter"
ylab="Flour"
main ="Contour plots of equal baking time"
N=25
colour.function = terrain.colors
H.grid <- seq(xlim[1], xlim[2], length = N)
V.grid <- seq(ylim[1], ylim[2], length = N)
grd <- expand.grid(H.grid, V.grid)
n <- dim(grd)[1]
valid.names <- colnames(model.frame(lsmodel))[dim(model.frame(lsmodel))[2]:2]
  valid.names <- valid.names[valid.names != xlab]
  valid.names <- valid.names[valid.names != ylab]
  h.points <- model.frame(lsmodel)[, xlab]
  v.points <- model.frame(lsmodel)[, ylab]
  expt_points <- data.frame(xlab = h.points, ylab = v.points)
  colnames(grd) <- c(xlab, ylab)
  colnames(expt_points) <- c(xlab, ylab)
  grd <- rbind(grd, expt_points)
  n_points_grid <- dim(grd)[1]
  for (elem in valid.names) {
    grd[[elem]] <- 0
  }
  grd$y <- predict(lsmodel, grd)
  binwidth <- (max(grd$y) - min(grd$y))/20
  p <- ggplot2::ggplot(data = grd[1:n, ], ggplot2::aes_string(x = xlab, 
                                                              y = ylab, z = "y")) + ggplot2::stat_contour(ggplot2::aes(color = ..level..), 
                                                                                                          binwidth = binwidth)   + 
    ggplot2::scale_colour_gradientn(colours = colour.function(N), name="Baking time (min)", limits = c(10, 30), breaks=c(10, 20, 30)) + 
    ggplot2::theme(panel.background = ggplot2::element_rect(fill = "white")) + 
    ggplot2::theme(panel.grid = ggplot2::element_blank()) + 
    ggplot2::theme_bw() + ggplot2::theme(plot.title = ggplot2::element_text(size = ggplot2::rel(2))) + 
    ggplot2::theme(axis.title = ggplot2::element_text(face = "bold", 
                                                      size = ggplot2::rel(1.5))) + ggplot2::labs(title = main) + 
    ggplot2::geom_point(data = grd[(n + 1):n_points_grid, 
    ], ggplot2::aes_string(x = xlab, y = ylab), size = 5) + 
    ggplot2::scale_x_continuous(breaks = seq(round(xlim[1]), 
                                             round(xlim[2]), by = 1)) + ggplot2::scale_y_continuous(breaks = seq(round(ylim[1]), 
                                                                                                                 round(ylim[2]), by = 1))
  bitmap('contour-plots-interaction-example.png', type="png256", width=9, height=9, res=300, pointsize=14)
  plot(p)
  dev.off()
  