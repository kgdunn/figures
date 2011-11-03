import Image
import numpy as np

# First run "process_flame_video.m" to generate the video files

import os
directory = 'C:\\kgd61600\\My-projects\\Figures\\examples\\flame\\'
for dirpath, dirnames, filenames in os.walk(directory):
	for filename in filenames:
		if filename.startswith('flame') and filename.endswith('.png'):
			filename
			im = Image.open(filename)

			# (left, upper, right, lower)
			box = (200, 350, im.size[0]-240, 800)
			im = im.crop(box)
			result = Image.new("RGBA", im.size)
			result.paste(im, (0, 0))
			result.save('chop-' + filename)
