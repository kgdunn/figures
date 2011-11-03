import Image
import numpy as np
import os

# First run "process_flame_video.m" to generate the video files

# import os
# directory = os.getcwd()
# for dirpath, dirnames, filenames in os.walk(directory):
#   for filename in filenames:
#       if filename.startswith('flame-') and filename.endswith('.png'):
#           filename
#           im = Image.open(filename)
# 
#           # (left, upper, right, lower)
#           box = (200, 350, im.size[0]-240, 800)
#           im = im.crop(box)
#           result = Image.new("RGBA", im.size)
#           result.paste(im, (0, 0))
#           result.save('chop-' + filename)
# 


video_framerate_per_second = 12
file_prefix = 'chop-flame-'
file_extension = 'png'
output_filename = 'flames.mp4'
command = ('ffmpeg -r ' + str(video_framerate_per_second) + ' -'
           'i ' + str(file_prefix) + '%03d.' + str(file_extension) + ' -y ' 
           '-an -b 2000k -minrate 2000k ' + str(output_filename))
f=os.popen(command)