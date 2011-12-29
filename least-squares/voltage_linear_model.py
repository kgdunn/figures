import numpy as np
from matplotlib.pyplot import *
 
x = np.array([0.01, 0.12, 0.24, 0.38, 0.51, 0.67, 0.84, 1.01, 1.15, 1.31])
y = np.array([273, 293, 313, 333, 353, 373, 393, 413, 433, 453])
n = np.max(x.shape)    # the number of observations
X = np.vstack([np.ones(n), x]).T
  
# Simpler, and more accurate way: 
a = np.linalg.lstsq(X, y)[0]
 
# Additional calculations
resids = y - np.dot(X,a)       # e = y - Xa; 
RSS = sum(resids**2)           # residual sum of squares
TSS = sum((y - np.mean(y))**2) # total sum of squares
R2 = 1 - RSS/TSS
std_error = np.sqrt(RSS/(n-len(a)))

# Plot the data along with the fitted line:
fig = figure()
plot(x, y, 'o', label='Original data', markersize=10)
plot(x, np.dot(X,a), 'r', label='Fitted line')
grid('on')
xlabel('Voltage [mV]')
ylabel('Temperature [K]')
legend(loc=0)

text(0.8, 325, 'Standard error = %0.1f K' % std_error)
plot(x, np.dot(X,a)+2*std_error, 'r--')
plot(x, np.dot(X,a)-2*std_error, 'r--')
fig.savefig('voltage-linear-model.png')
