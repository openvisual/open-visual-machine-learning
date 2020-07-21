# -*- coding: utf-8 -*-

# 변경 사항
# 함수 모듈화
# 히스토그램 정규화 추가

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, cv2, numpy as np, sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
print( "Done Import.".center( 80, "*") )

# 현재 파일의 폴더로 실행 폴더를 이동함.
print( "Pwd 1: %s" % os.getcwd())
# change working dir to current file
dirname = os.path.dirname(__file__)
if dirname :
    os.chdir( dirname )
    print( "Pwd 2: %s" % os.getcwd())
pass
#-- 현재 파일의 폴더로 실행 폴더를 이동함.

#TODO     이미지 저장 함수
img_save_cnt = 0

def img_file_name( work ) :
    global img_save_cnt
    img_save_cnt += 1

    fn = img_path

    root = fn[ : fn.rfind( "/") ]

    folder = root + "/temp"

    if os.path.exists( folder ):
        if not os.path.isdir( folder ) :
            os.remove( folder )
            os.mkdir( folder )
        else :
            # do nothing
            pass
        pass
    else :
        os.mkdir( folder )
    pass

    fn = fn.replace( root , "" )
    k = fn.rfind( "." )
    fn = folder + fn[ : k ] + ( "_%02d_" % img_save_cnt) + work + fn[k:]
    return fn
pass #-- img_file_name

def save_img_as_file( work, img, cmap="gray"):
    fn = img_file_name( work )
    plt.imsave( fn, img, cmap='gray' )

    log.info( "Image saved as file name[%s]" % fn )
pass #-- save_img_as_file

# -- 이미지 저장 함수

# pyplot ax 의 프레임 경계 색상 변경
def change_ax_border_color( ax, color ) :
    for spine in ax.spines.values():
        spine.set_edgecolor( color )
    pass
pass #-- change_ax_border_color

#TODO    원천 이미지 획득

# 이미지를 파일로 부터 RGB 색상으로 읽어들인다.
img_path = "../data_ocr/sample_01/sample_21.png"
img_path = "../data_ocr/sample_01/hist_work_01.png"
img_path = "../data_ocr/sample_01/messi5.png"

img_org = cv2.imread( img_path, cv2.IMREAD_COLOR ) #BGR order

# 이미지 높이, 넓이, 채널수 획득
height      = img_org.shape[0]
width       = img_org.shape[1]
channel_no  = img_org.shape[2]

print( "Image path: %s" % img_path )
print( "Image widh: %s, height: %s, channel: %s" % (width,height,channel_no ) )

save_img_as_file( "org", img_org, cmap="rgb" )

fig = plt.figure(figsize=(10, 10), constrained_layout=True)

# org img, channel img, gray scale, median blur, histogram, bin, y_count
gs_row_cnt = 7
gs_col_cnt = 3

gs_row = -1
gs_col = 0

gridSpec = GridSpec( gs_row_cnt, gs_col_cnt, figure=fig )

if 1 : # 원본 이미지 표출
    gs_row += 1
    gs_col = 0
    colspan = gs_col_cnt
    img = img_org

    title = 'Original Image: %s' % ( img_path.split("/")[-1] )

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))
    img_show = ax.imshow( img )
    ax.set_xlabel( 'x\n %s' % title )
    ax.set_ylabel( 'y', rotation=0 )

    change_ax_border_color( ax, "green" )

    fig.colorbar(img_show, ax=ax)
pass

#-- 원천 이미지 획득

#TODO   채널 분리
# b, g, r 채널 획득
# cv2.imread() 는 b, g, r 순서대로 배열에서 반환한다.
b_channel = img_org[:,:,0].copy()
g_channel = img_org[:,:,1].copy()
r_channel = img_org[:,:,2].copy()

channels = [ r_channel, g_channel, b_channel ]

if 0 :  # 채널 이미지 표출
    gs_row += 1
    gs_col = 0
    colspan = 1

    for i, channel in enumerate( channels ):
        img_temp = np.zeros( (height, width, 3), dtype='uint8' )
        img_temp[ :, : , i ] = channel
        img = img_temp

        ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))
        img_show = ax.imshow( img )
        ax.set_ylabel( 'y', rotation=0 )

        title = "red channel"
        if i == 1 :
            title = "green channel"
        elif i == 2 :
            title = "blue channel"
        pass

        ax.set_xlabel( 'x\n%s' % title )

        ( i == 2 ) and fig.colorbar(img_show, ax=ax)

        gs_col += colspan
    pass
pass

#-- 채널 분리

#TODO    Grayscale 변환

# grayscale 변환 함수
def convert_to_grayscale( channels ) :
    log.info( "convert to grayscale...." )

    r_channel = channels[ 0 ]
    g_channel = channels[ 1 ]
    b_channel = channels[ 1 ]

    h = len( r_channel ) # image height
    w = len( r_channel[0] ) # image width

    data = np.empty( ( h, w ), dtype='f')

    for y in range( h ) :
        for x in range( w ) :
            # RGB -> GrayScale 변환 공식
            # average  Y = (R + G + B / 3)
            # weighted Y = (0.3 * R) + (0.59 * G) + (0.11 * B)
            # Colorimetric conversion Y = 0.2126R + 0.7152G  0.0722B
            # OpenCV CCIR Y = 0.299 R + 0.587 G + 0.114 B
            gs = 0.299*r_channel[y][x] + 0.587*g_channel[y][x] + 0.114*b_channel[y][x]
            data[y][x] = gs
        pass
    pass

    return data
pass
# -- grayscale 변환

#TODO   image reverse 변환 함수
def reverse_image( image, max ) :
    log.info( "reverse image...." )

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.empty( ( h, w ), dtype='f')

    for y in range( h ) :
        for x in range( w ) :
            v = image[y][x]
            data[y][x] = max - v
        pass
    pass

    return data
pass
# -- image reverse 변환

# grayscale 변환
grayscale = convert_to_grayscale( channels )
# 영상 역전
grayscale = reverse_image( grayscale, max = 255 )

save_img_as_file( "grayscale", grayscale )

if 1 : # 그레이 스케일 이미지 표출
    gs_row += 1
    gs_col = 1
    colspan = gs_col_cnt - gs_col
    img = grayscale
    cmap = "gray"
    title = "Grayscale"

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    img_show = ax.imshow( img, cmap=cmap )
    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'y', rotation=0 )

    change_ax_border_color( ax, "green" )

    fig.colorbar(img_show, ax=ax)
pass #-- 그레이 스케일 이미지 표출

gs_avg = np.average( grayscale )
gs_std = np.std( grayscale )
sg_max = np.max( grayscale )

log.info( "grayscale avg = %s, std = %s" % (gs_avg, gs_std))
#-- grayscale 변환

#TODO     Grayscale histogram 생성

# calculate histogram count
def make_histogram( grayscale ) :
    log.info( "Make histogram ..." )

    histogram = np.zeros( 256, dtype='u8' )

    for _, row in enumerate( grayscale ) :
        for x, gs in enumerate( row ) :
            gs = (int)( gs )
            histogram[ gs ] += 1
        pass
    pass

    return histogram
pass #-- calculate histogram

#TODO    누적 히스토 그램
def accumulate_histogram( image ) :
    log.info( "accumulate histogram" )

    sum = 0

    data = np.empty( len( image ), dtype='u8' )
    for x, v in enumerate( image ) :
        sum += v
        data[x] = sum
    pass

    return data
pass # 누적 히스트 그램

histogram = make_histogram( grayscale )
histogram_acc = accumulate_histogram( histogram )

def show_histogram( histogram , histogram_acc, title ): # 히스토 그램 표출
    #gs_row += 1
    gs_col = 0
    colspan = 1

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    charts = { }

    hist_avg = np.average( histogram )
    hist_std = np.std( histogram )
    hist_max = np.max( histogram )

    log.info( "hist avg = %s, std = %s" % (hist_avg, hist_std) )

    if histogram_acc is not None :
        # accumulated histogram
        y = histogram_acc
        x = [i for i, _ in enumerate( y ) ]
        charts["accumulated"] = ax.bar( x, y, width=0.4, color='yellow', alpha=0.3 )
    pass

    # histogram bar chart
    y = histogram
    x = [i for i, _ in enumerate( y ) ]
    charts["count"] = ax.bar( x, y, width=0.5, color='red' )

    # histogram std chart
    x = [ gs_avg - gs_std, gs_avg + gs_std ]
    y = [ hist_max*0.95, hist_max*0.95 ]
    charts["std"] = ax.fill_between( x, y, color='cyan', alpha=0.5 )

    # histogram average chart
    x = [ gs_avg, ]
    y = [ hist_max, ]
    charts["average"] = ax.bar(x, y, width=0.5, color='blue', alpha=0.5)

    if 1 : # 레전드 표출
        t = [ ]
        l = list( charts.keys() )
        l = sorted( l )
        for k in l :
            t.append( charts[ k ] )
        pass

        for i, s in enumerate( l ) :
            import re
            s = s[0] + re.sub(r'[aeiou]', '', s[1:])
            l[i] = s[:4]
        pass

        loc = "upper right"

        if gs_avg > 122 :
            loc = "upper left"
        pass

        ax.legend( t, l, loc=loc, shadow=True)
    pass #-- 레전드 표출

    if 1 : # x 축 최대, 최소 설정
        max_x = gs_avg + gs_std*1.2

        ax.set_xlim( 0, max_x )
    pass

    ax.set_xlabel( title )
    ax.set_ylabel( 'Count', rotation=90 )
pass #-- 히스토 그램 표출

if 1 :
    show_histogram( histogram, histogram_acc, title = "Grayscale Histogram" )
pass

#-- histogram 생성

#TODO    히스토그램 평활화

def normalize_image_by_histogram( image, histogram_acc ) :
    log.info( "Normalize histogram")

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.empty( [h, w], dtype=image[0].dtype )

    # https://en.wikipedia.org/wiki/Histogram_equalization
    N = h*w # pixel count
    Lmax = np.max( image ) # max pixel value

    cdf = histogram_acc
    cdf_min = np.min( np.nonzero(cdf) )

    for y, row in enumerate( image ):
        for x, v in enumerate( row ):
            v = int( v )
            v = (cdf[v] - cdf_min)/(N - cdf_min)*Lmax
            v = round( v )
            data[y][x] = v
        pass
    pass

    return data
pass #-- normalize_image_by_histogram

image_normalized = normalize_image_by_histogram( grayscale, histogram_acc )

save_img_as_file( "image_normalized", image_normalized )

if 1 : # 평활화 이미지 표출
    gs_row += 1
    gs_col = 0
    colspan = gs_col_cnt
    title = "Normalization"
    img = image_normalized
    cmap = "gray"

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))
    img_show = ax.imshow( img, cmap=cmap )

    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'y', rotation=0 )

    change_ax_border_color( ax, "green" )

    fig.colorbar(img_show, ax=ax)
pass #-- 평활화 이미지 표출

#-- 히스토그램 평활화

#TODO   잡음 제거
# Median Blur Filter 적용

# 잡음 제거 함수
def remove_noise( image ) :
    log.info( "remove noise...." )

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.empty( [h, w], dtype='f')

    ksize = 5

    for y in range( height ) :
        for x in range( width ) :
            window = image[ y : y + ksize, x : x + ksize ]
            median = np.median( window )
            data[y][x] = median
        pass
    pass

    return data
pass #-- 잡음 제거 함수

noise_removed = remove_noise( image_normalized )

save_img_as_file( "noise_removed", noise_removed )

if 1 : # 잡음 제거  이미지 표출
    gs_row += 1
    gs_col = 1
    colspan = gs_col_cnt - gs_col
    img = noise_removed
    cmap = "gray"
    title = "Noise removed (Median Blur)"

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    img_show = ax.imshow( img, cmap=cmap )
    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'y', rotation=0 )

    change_ax_border_color( ax, "blue" )

    fig.colorbar(img_show, ax=ax)
pass #-- 잡음 제거  이미지 표출

#-- 잡음 제거를 위한 Median Blur Filter

#TODO    이진화

# 이진화 계산
def binarize_image( image, threshold = None ):
    log.info( "Binarize threshold" )

    if not threshold :
        threshold= np.average( image )
    pass

    log.info( "Threshold = %s" % threshold )

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.empty( ( h, w ), dtype='B')

    for y, row in enumerate( image ) :
        for x, gs in enumerate( row ) :
            gs = round( gs ) # 반올림.
            data[y][x] = (0, 1,)[ gs >= threshold ]
        pass
    pass

    return data, threshold
pass # -- 이진화 계산

target_image = noise_removed
image_binarized, threshold = binarize_image( image = target_image )

save_img_as_file( "image_binarized", image_binarized )

if 1 : # 이진 이미지 표출
    gs_row += 1
    gs_col = 0
    colspan = gs_col_cnt
    title = "Binarization (threshold=%s)" % threshold
    img = image_binarized
    cmap = "gray"

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))
    img_show = ax.imshow( img, cmap=cmap )

    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'y', rotation=0 )

    change_ax_border_color( ax, "green" )

    fig.colorbar(img_show, ax=ax)
pass #-- 이진 이미지 표출

#-- 이진화

#TODO   Y 축 데이터 히스토그램

def count_y_axis_signal( image, ksize ) :
    log.info( "y axis signal count" )

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.zeros( [w], dtype='B')
    ksize = 1

    for x in range( width ) :
        window = image[ 0 : height , x : x + ksize ]
        signal_count = np.count_nonzero( window == 1 ) # 검정 바탕 흰색 카운트
        # signal_count = np.count_nonzero( window == 0 ) # 흰 바탕 검정 카운트
        data[x] = signal_count
    pass

    return data
pass #-- count_y_axis_signal

target_image = image_binarized
y_counts = count_y_axis_signal( image= target_image, ksize = 1 )

if 1 : # y count 표출
    gs_row += 1
    gs_col = 0
    colspan = gs_col_cnt

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    # 이진 이미지 표출
    img = image_binarized
    cmap = "gray"
    img_show = ax.imshow( img, cmap=cmap )
    fig.colorbar(img_show, ax=ax)
    #-- 이진 이미지 표출

    charts = { }

    # y count bar chart
    y = y_counts
    x = [i for i, _ in enumerate( y ) ]
    charts["y count"] = ax.bar( x, y, width=0.5, color='red', align='center', alpha=1.0)

    if 0 : # 레전드 표출
        t = [ ]
        l = list( charts.keys() )
        l = sorted( l )
        for k in l :
            t.append( charts[ k ] )
        pass
        loc = "upper right"
        ax.legend( t, l, loc=loc, shadow=True)
    pass #-- 레전드 표출

    title = "y count"
    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'Y count', rotation=90 )

    ax.set_xlim( 0, width )
pass #-- y count 표출

#-- y count 표출

plt.show()

# end