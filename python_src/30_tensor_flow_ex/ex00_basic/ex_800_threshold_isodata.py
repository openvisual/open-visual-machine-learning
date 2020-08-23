# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings 
#warnings.filterwarnings('ignore',category=FutureWarning)

import logging as log
log.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO
    )

import os, cv2, numpy as np, sys

import matplotlib as mpl
import matplotlib.pyplot as plt

# hide toolbar
mpl.rcParams['toolbar'] = 'None'

file_path = "../data_opencv/messi5.jpg"

grayscale = cv2.imread( file_path, 0 ) # read image as grayscale

# 히스토 그램 계산
histogram = [0]*256

h = len( grayscale )
w = len( grayscale[0] )
for row in grayscale :
    for x in row :
        histogram[x] += 1
    pass
pass
# histogram 계산

ax = plt.subplot( 3, 1, 1 )

ax.imshow( grayscale, cmap="gray" )
# 히스토그램 스케일 출력
ax.bar( range(256) , h*(histogram/np.max( histogram )), color="y", label="histogram")

ax.set_xlabel( "Grayscale" )
ax.set_ylabel("y")
ax.legend()

thresh_avg = sum( histogram * np.arange(256) )/(w*h)

print( "thresh_avg = " , thresh_avg )
bin_by_threshold_avg = np.where( grayscale >= thresh_avg , 1, 0 )
ax = plt.subplot( 3, 1, 2 )
ax.imshow( bin_by_threshold_avg, cmap="gray" )
ax.set_xlabel( f"Threshold average({thresh_avg:.2f})" )
ax.set_ylabel("y")

# isodata 임계치 구하기
t = 0
t_diff = None
for i in range( 2, 256 ) :
    mL_hist = histogram[ 0 : i ]
    mH_hist = histogram[ i : ]
    mL = sum(mL_hist * np.arange(0, i)) / sum(mL_hist)
    mH = sum(mH_hist * np.arange(i, 256)) / sum(mH_hist)

    diff = abs( i - (mL + mH)/2 )
    if t_diff is None or diff < t_diff :
        t_diff = diff
        t = i
    pass
pass

bin_by_threshold_isodata = np.where( grayscale >= t , 1, 0 )
ax = plt.subplot( 3, 1, 3 )
ax.imshow( bin_by_threshold_isodata, cmap="gray" )
ax.set_xlabel( f"Threshold isodata({t})" )
ax.set_ylabel("y")

plt.show()