import cv2
import numpy as np
from PIL import Image
import os
import sys
import shutil

#import image
imageName=sys.argv[1]
image = cv2.imread(imageName);
image= cv2.resize(image,(500,500));

#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second',thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((1,1), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
j=0
dir='./SegmentedCharsOf-'+imageName+'/';
if not os.path.exists(dir):
    os.makedirs(dir)
else:
    shutil.rmtree(dir)           #removes all the subdirectories!
    os.makedirs(dir)
os.chdir(dir);
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
	
    # Getting ROI
    roi = image[y:y+h, x:x+w];roi= cv2.resize(roi,(28,28));
    if(h>20 or w>20 and h<100 and w<100): cv2.imwrite(str(j)+'.jpg',roi); j=j+1; print("\n Subimages of characters saved");

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    if(h>20 or w>20 and h<100 and w<100): cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.waitKey(0)
	
#cv2.imshow('marked areas',image)
cv2.waitKey(0)
