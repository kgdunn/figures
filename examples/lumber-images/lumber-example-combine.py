import Image
import numpy as np

# Plots extracted from MACCMIA's right-click feature

chop = 110
left = Image.open('lumber-example-original-image.png')
mid = Image.open('score-image-t1.png')
right = Image.open('residual-image-1.png')
box = (chop, 0, left.size[0]-chop, left.size[1]) # (left, upper, right, lower)
left = left.crop(box)
mid = mid.crop(box)
right = right.crop(box)
width = left.size[0] + right.size[0] + mid.size[0]
height = max(left.size[1], right.size[1])
result = Image.new("RGBA", (width, height))
result.paste(left, (0, 0))
result.paste(mid, (left.size[0], 0))
result.paste(right, (left.size[0]+mid.size[0], 0))
result.save('component-1.png')


left = Image.open('lumber-example-SPE-after-one-PC.png')
mid = Image.open('score-image-t2.png')
right = Image.open('residual-image-2.png')
box = (chop, 0, left.size[0]-chop, left.size[1]) # (left, upper, right, lower)
left = left.crop(box)
mid = mid.crop(box)
right = right.crop(box)
width = left.size[0] + right.size[0] + mid.size[0]
height = max(left.size[1], right.size[1])
result = Image.new("RGBA", (width, height))
result.paste(left, (0, 0))
result.paste(mid, (left.size[0], 0))
result.paste(right, (left.size[0]+mid.size[0], 0))
result.save('component-2.png')
