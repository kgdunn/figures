

# Exported from ProMV version 11.08
data = [\
['Tinned fruit', 1, 1.31654998477158, 0.861118461287433, 1.77198150825572],
['Frozen veggies', 2, 1.26764598018534, 0.633109200585117, 1.90218275978556],
['Instant coffee', 3, 1.24552865380916, 0.667943212203045, 1.82311409541527],
['Crisp bread', 4, 1.2230185023799, 0.411758734728515, 2.03427827003128],
['Frozen fish', 5, 1.19258081628244, 0.128511903569881, 2.25664972899499],
['Sweetener', 6, 1.16357010539303, 0.788208063430221, 1.53893214735584],
['Powder soup', 7, 1.13152980730308, 0.864503487558439, 1.39855612704772],
['Tin soup', 8, 1.09856438614819, 0.41256792781124, 1.78456084448514],
['Apples', 9, 1.09294749483652, 0.293526328135918, 1.89236866153713],
['Garlic', 10, 1.01626233140326, 0.404674727491952, 1.62784993531457],
['Tea', 11, 0.996968115718746, 0.732839806288485, 1.26109642514901],
['Jam', 12, 0.970270938667419, 0.259641753881137, 1.6809001234537],
['Biscuits', 13, 0.940406337551399, 0.518737434112078, 1.36207524099072],
['Yoghurt', 14, 0.937142027800112, -0.177709687736681, 2.05199374333691],
['Potatoes', 15, 0.845300058166701, 0.0905875712504337, 1.60001254508297],
['Oranges', 16, 0.817981868705577, 0.0934787800383058, 1.54248495737285],
['Olive oil', 17, 0.630016393948752, -0.039555088688831, 1.29958787658634],
['Real coffee', 18, 0.529227726690094, -0.509916760779205, 1.56837221415939],
['Butter', 19, 0.434900410099994, -0.484524141396117, 1.35432496159611],
['Margarine', 20, 0.417426456115651, -0.458383603593326, 1.29323651582463],]

import matplotlib.pyplot as plt
import numpy as np
data = np.array(data)

value = np.array([float(v) for v in data[:,2]])
LB = np.array([float(v) for v in data[:,3]])
UB = np.array([float(v) for v in data[:,4]])
errors = UB - value # same as value-LB
import numpy as np

# 
# N = 5
# menMeans   = (20, 35, 30, 35, 27)
# womenMeans = (25, 32, 34, 20, 25)
# menStd     = (2, 3, 4, 1, 2)
# womenStd   = (3, 5, 2, 3, 3)
# 
width = 0.45       # the width of the bars: can also be len(x) sequence
N = data.shape[0]
index = np.arange(N)

fig = plt.figure(figsize=(9, 6))
rect = [0.1, 0.24, 0.85, 0.7]  # Left, bottom, width, height
ax = fig.add_axes(rect, frame_on=True)
ax.bar(index, value,  width, color='green', yerr=errors, ecolor='k', capsize=4)
ax.set_ylabel('VIP values', weight='bold')
ax.set_title('VIP after 2 components')
ax.set_xticklabels(index+width/2)
for label in ax.get_xticklabels():
    label.set_weight('bold')
ax.set_ylim([-0.6, 2.4])   
ax.set_xlim([-0.5, N-0.5])   
plt.xticks(index+width/2, data[:,0], rotation=90)
ax.grid(True)
fig.savefig('VIP-plot-demo.png', dpi=150, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=True)

