import numpy as np
import matplotlib.pyplot as plt

wavelengths = np.arange(1100, 2498+0.01, 2)
X = np.loadtxt('corn.csv', delimiter=',')
X = X[:, 0:700]


fig = plt.figure(figsize=(9, 5))  # Use (8, 5.5) for slide aspect ratio
rect = [0.1, 0.15, 0.89, 0.75]  # Left, bottom, width, height
ax = fig.add_axes(rect, frame_on=True)
for loc, spine in ax.spines.iteritems():
    if loc in ['left','bottom']:
        spine.set_position(('outward',10)) # outward by 10 points
    elif loc in ['right','top']:
        spine.set_color('none') # don't draw spine
    
# turn off ticks where there is no spine
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


# Add labels
ax.set_xlabel('Wavelength (nm)', fontsize=16)
ax.set_ylabel('NIR response', fontsize=16)
ax.set_title(r'Spectra of 80 corn samples', fontsize=16, fontweight='bold')
ax.patch.set_facecolor('None')

for spectrum in X:
     ax.plot(wavelengths, spectrum)
     ax.hold(True)
     
for label in ax.get_xticklabels():
    label.set_fontsize(16)
    label.set_weight(800)
for label in ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_weight(800)


ax.axis([1100, 2500, 0, 1])
ax.grid(True)

fig.savefig('corn-spectra.png', dpi=150, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=True)
