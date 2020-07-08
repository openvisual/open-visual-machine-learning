# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
print( "Done Import.".center( 80, "*") )

print( "Pwd 1: %s" % os.getcwd())
# change working dir to current file
os.chdir(os.path.dirname(__file__))
print( "Pwd 2: %s" % os.getcwd())

# 원천 이미지 획득
# 이미지를 파일로 부터 RGB 색상으로 읽어들인다.
img_path = '../data_opencv_sample/messi5.jpg'
#img_path = "../data_ocr/sample_1.png"

img = cv2.imread( img_path, cv2.IMREAD_COLOR ) #BGR order

# 이미지 높이, 넓이, 채널수 획득 
height      = img.shape[0]
width       = img.shape[1]
channel_no  = img.shape[2]

print( "Image path: %s" % img_path )
print( "Image widh: %s, height: %s, channel: %s" % (width,height,channel_no ) )

fig = plt.figure(figsize=(10, 10), constrained_layout=True)
gs_row_cnt = 5 # org img, channel img, gray scale, histogram, bin
gs_col_cnt = 4

gs_row = -1 
gs_col = 0 

gs = GridSpec( gs_row_cnt, gs_col_cnt, figure=fig )

if 0 :
    plt.imshow(img)
pass

if 1 :
    gs_row += 1 
    gs_col = 0 
    ax = plt.subplot(gs.new_subplotspec((gs_row, gs_col), colspan=3))
    ax.imshow( img )
    ax.set_xlabel( 'x' )
    ax.set_ylabel( 'y' )
    
    ax.set_title( 'Original Image: %s' % ( img_path.split("/")[-1] ) ) 

    0 and plt.show() 
pass
#-- 원천 이미지 획득

# 채널 분리 
# b, g, r 채널 획득
# cv2.imread() 는 b, g, r 순서대로 배열에서 반환한다.
b_channel = img[:,:,0].copy() 
g_channel = img[:,:,1].copy()
r_channel = img[:,:,2].copy()

channels = [ r_channel, g_channel, b_channel ]

if 0 : 
    plt.figure()
    for i, channel in enumerate( channels ):
        img_temp = np.zeros( (height, width, 3), dtype='uint8' ) 
        img_temp[ :, : , i ] = channel 
        plt.subplot( 1, 3, i + 1  ) 
        plt.grid(False)
        plt.imshow( img_temp ) 

        label = "red"
        if i == 1 :
            label = "green"
        elif i == 2 :
            label = "blue"
        pass
        plt.xlabel( label )
    pass

    plt.show()
pass

if 1 :
    gs_row += 1 
    gs_col = 0 
    ax = plt.subplot(gs.new_subplotspec((gs_row, gs_col), colspan=1))
    ax.imshow( img )
    ax.set_xlabel( 'x' )
    ax.set_ylabel( 'y' )
    
    ax.set_title( 'Original Image: %s' % ( img_path.split("/")[-1] ) ) 

    plt.show() 
pass

#-- 채널 분리 

# RGB -> GrayScale 변환 공식
print( "Grayscale" )
gray_scale = np.empty( ( height, width ), dtype='uint8') 

for y, row in enumerate( gray_scale ) :
    for x, _ in enumerate( row ) :
        # average Y = (R + G + B / 3)
        # weighted Y = (0.3 * R) + (0.59 * G) + (0.11 * B)
        # Colorimetric conversion Y = 0.2126R + 0.7152G  0.0722B
        # OpenCV CCIR Y = 0.299 R + 0.587 G + 0.114 B
        gs = 0.299*r_channel[y][x] + 0.587*g_channel[y][x] + 0.114*b_channel[y][x]
        gs = (int)(round(gs))
        gray_scale[y][x] = gs
    pass
pass

if 0 : 
    #print( gray )
    #plt.imshow( gray, cmap='gray', vmin=0, vmax=255)
    plt.imshow( gray_scale, cmap='gray' )
    plt.title( "GrapyScale" )
    plt.colorbar()
    plt.show()
pass

gs_avg = np.average( gray_scale )
gs_std = np.std( gray_scale )
sg_max = np.max( gray_scale )

print( "grayscale avg = %s, std = %s" % (gs_avg, gs_std))

# histogram 생성 
print( "hostogram" )
# calculate histogram count
histogram = np.zeros( 256, dtype=float )

for y, row in enumerate( gray_scale ) : 
    for x, gs in enumerate( row ) :
        histogram[ gs ] += 1
    pass
pass
#-- calculate histogram

hist_avg = np.average( histogram )
hist_std = np.std( histogram )
hist_max = np.max( histogram )

print( "hist avg = %s, std = %s" % (hist_avg, hist_std))

if 1 :
    fig, ax = plt.subplots()

    charts = { }
    
    y = histogram
    x = [i for i, _ in enumerate(histogram) ]
    charts[ "count" ] = ax.bar( x, y, width=0.5, color='green', align='center', alpha=1.0)

    x = [gs_avg - gs_std, gs_avg + gs_std]
    y = [ hist_max*0.95, hist_max*0.95 ]
    charts[ "std"] = ax.fill_between( x, y, color='cyan', alpha=0.5 ) 
    
    x = [ gs_avg, ]
    y = [ hist_max, ]
    charts[ "average" ] = ax.bar(x, y, width=1, color='blue', align='center', alpha=0.5) 

    loc = "upper right"

    if gs_avg > 122 :
        loc = "upper left"
    pass

    ax.legend( charts, loc=loc, shadow=True)

    ax.set_xlabel( 'GrayScale' )
    ax.set_ylabel( 'Count' )
    
    ax.set_title( 'Histogram' ) 

    min_x = gs_avg - gs_std*1.2

    if min_x < 0 :
        min_x = 0 
    pass

    ax.set_xlim( min_x, 255 )

    plt.show()
pass 
#-- histogram 생성 

# 이진화

print( "binarization" )

threshold = gs_avg

print( "threshold: %s" % threshold )

bin = np.empty( ( height, width ), dtype='B')
for y, row in enumerate( gray_scale ) :
    for x, gs in enumerate( row ) :
        bin[y][x] = (0, 1)[ gs >= threshold ]
    pass
pass

if 1 : 
    print( bin )
    plt.imshow( bin, cmap='binary' )
    plt.title( "Binarization (threshold=%s)" % threshold )
    plt.colorbar()
    plt.show()
pass

# -- 이진화 

if 0 : 
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
pass