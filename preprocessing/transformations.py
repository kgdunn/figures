import numpy as np
from matplotlib.pylab import plot

x1 = np.arange(1, 6, 0.05)
x2 = 5 + 2*x1 +8*x1**2
x1pp = (x1 - np.mean(x1))/np.std(x1)
x2pp = (x2 - np.mean(x2))/np.std(x2)

x2tr = np.sqrt(x2)
x2tr_pp = (x2tr - np.mean(x2tr))/np.std(x2tr)

plot(x1pp, x2pp)
plot(x1pp, x2tr_pp)
