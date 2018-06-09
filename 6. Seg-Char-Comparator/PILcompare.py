from PIL import Image as i
from PIL import ImageChops as ic
import sys
import math

im1=i.open(sys.argv[1])
im2=i.open(sys.argv[2])
diff = ic.difference(im1, im2)
h = diff.histogram()
sq = (value*((idx%256)**2) for idx, value in enumerate(h))
sum_of_squares = sum(sq)
rms = math.sqrt(sum_of_squares / float(im1.size[0] * im1.size[1]))

print('comparing '+str(sys.argv[1])+' and '+str(sys.argv[2])+' RMSE obtained is : '+str(rms))
if(rms<90):
	print('wow quiet good')