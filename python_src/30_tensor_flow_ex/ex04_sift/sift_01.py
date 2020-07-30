# coding: utf-8

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

print( "OpenCV version : %s" % cv.__version__ )

print( "Pwd 1: %s" % os.getcwd())
# change working dir to current file
dirname = os.path.dirname(__file__)
dirname and os.chdir( dirname )
dirname and print( "Pwd 2: %s" % os.getcwd())

img = cv.imread('../data_opencv/home.jpg')

gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()

kp = sift.detect(gray, None)

print( "keypoints: " , kp )

img = cv.drawKeypoints( gray, kp, img )

cv.imwrite('sift_keypoints.jpg',img)

cv.imshow( "SIFT", img )

cv.waitKey(0)  
  
cv.destroyAllWindows() 

# end