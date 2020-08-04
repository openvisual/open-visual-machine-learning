# -*- coding: utf-8 -*-

'''
# 변경 사항
# -- 함수 모듈화
# -- 히스토그램 정규화 추가
# -- 결과 이미지 저장
# -- Adaptive Thresholding 추가
# -- 클래스 화
# 축소/팽창
# histogram modal count 계산
# OTSU thresholding
# ROSIN thresholding
#
# 라인 추출
'''

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

''' profile functions '''

# -- usage
# @profile
# def your_function(...):
#       ....
#
# your_function( ... )
# print_prof_data()

import time
from functools import wraps

PROF_DATA = {}
PROF_LAST = None

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        pass

        global PROF_LAST
        PROF_LAST = fn.__name__

        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling
pass # -- profile(fn)

def print_prof_name(fn_name):
    data = PROF_DATA[ fn_name ]

    max_time = max(data[1])
    avg_time = sum(data[1]) / len(data[1])
    msg = f"*** The function[{fn_name}] was called {data[0]} times. Exe. time max: {max_time:.3f}, average: {avg_time:.3f}"
    log.info( msg )
pass # -- print_prof_name()

def print_prof_last( ) :
    PROF_LAST and print_prof_name( PROF_LAST )
pass

def print_prof_data():
    for fn_name in PROF_DATA :
        print_prof_name( fn_name )
    pass
pass # -- print_prof_data()

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
pass # -- clear_prof_data

''' --- profile functions '''

def explorer_open( path ) :
    ''' open folder by an explorer'''
    import webbrowser as wb
    wb.open( path )
pass # -- open_folder

def file_name_except_path_ext(path):
    # 확장자와 파일 패스를 제외한 파일 이름 구하기.
    head, file_name = os.path.split(path)

    dot_idx = file_name.rfind(".")
    file_name = file_name[: dot_idx]

    return file_name
pass # file_name_except_path_ext


def to_excel_letter(col):
    excel_column = ""

    AZ_len = ord('Z') - ord('A') + 1

    def to_alhapet(num):
        c = chr(ord('A') + int(num))
        return c

    pass  # -- to_alhapet

    while col > 0:
        col, remainder = divmod(col - 1, AZ_len)

        excel_column = to_alhapet(remainder) + excel_column
    pass

    if not excel_column:
        excel_column = "A"
    pass

    return excel_column
pass  # -- to_excel_letter


def remove_space_except_first(s):
    # 첫 글자를 제외한 나머지 모음을 삭제한다.
    import re
    reg_exp = r'[aeiou]'
    s = s[0] + re.reg_str(reg_exp, '', s[1:])

    return s
pass # -- remove_space_except_first

import os, cv2, numpy as np, sys, time
import math
from math import pi

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
log.info( "Done Import.".center( 80, "*") )

# 현재 파일의 폴더로 실행 폴더를 이동함.
log.info( f"Pwd 1: {os.getcwd()}" )
dirname = os.path.dirname(__file__) # change working dir to current file
if dirname :
    os.chdir( dirname )
    log.info(f"Pwd 2: {os.getcwd()}")
pass
#-- 현재 파일의 폴더로 실행 폴더를 이동함.

# 이미지를 파일로 부터 RGB 색상으로 읽어들인다.
#img_path = "../data_ocr/sample_01/messi5.png"
#img_path = "../data_ocr/sample_01/hist_work_01.png"
#img_path = "../data_ocr/sample_01/gosu_01.png"
img_path = "../data_ocr/sample_01/sample_21.png"
#img_path = "../data_yegan/ex_01/_1018877.JPG"
#img_path = "../data_yegan/ex_01/1-56.JPG"

#TODO    원천 이미지 획득

img_org = cv2.imread( img_path, cv2.IMREAD_COLOR ) #BGR order

# 이미지 높이, 넓이, 채널수 획득
height      = img_org.shape[0]
width       = img_org.shape[1]
channel_no  = img_org.shape[2]

log.info( f"Image path: {img_path}"  )
print( f"Image widh: {width}, height: {height}, channel: {channel_no}" )

fig = plt.figure(figsize=(13, 10), constrained_layout=True)
plt.get_current_fig_manager().canvas.set_window_title("2D Line Extraction")

# org img, channel img, gray scale, median blur, histogram, bin, y_count
gs_row_cnt = 5
gs_col_cnt = 7

gs_row = -1
gs_col = 0

gridSpec = GridSpec( gs_row_cnt, gs_col_cnt, figure=fig )

#-- 원천 이미지 획득

class Gap :
    # a gap between segments
    def __init__(self, coord):
        self.coord = coord
    pass

    def distance(self):
        coord = self.coord

        diff = coord[1] - coord[0]
        return abs( diff )
    pass
pass # -- Gap

class SegInfo :
    # segmentation info
    def __init__(self, coord):
        self.coord = coord
    pass

    def distance(self):
        coord = self.coord

        diff = coord[1] - coord[0]
        return diff
    pass
pass # -- Segment

class Image :

    def __init__(self, img, algorithm="" ):
        # 2차원 배열 데이터
        self.img = img
        self.algorithm = algorithm
        self.histogram = None
        self.histogram_acc = None
    pass

    # TODO     이미지 저장 함수
    img_save_cnt = 0

    def img_file_name(self, work):
        # C:/temp 폴더에 결과 파일을 저정합니다.

        Image.img_save_cnt += 1

        img_save_cnt = Image.img_save_cnt

        fn = img_path

        root = fn[: fn.rfind("/")]

        folder = "C:/temp"

        if os.path.exists(folder):
            if not os.path.isdir(folder):
                os.remove(folder)
                os.mkdir(folder)
            else:
                # do nothing
                pass
            pass
        else:
            os.mkdir(folder)
        pass

        fn = fn.replace(root, "")
        k = fn.rfind(".")
        fn = folder + fn[: k] + ("_%02d_" % img_save_cnt) + work + fn[k:]
        return fn

    pass  # -- img_file_name

    def save_img_as_file(self, work, cmap="gray"):
        filename = self.img_file_name(work)
        img = self.img

        plt.imsave(filename, img, cmap=cmap)

        log.info( f"Image saved as file name[ {filename} ]" )
    pass  # -- save_img_as_file

    ''' -- 이미지 저장 함수 '''

    # TODO 플롯 함수
    # pyplot ax 의 프레임 경계 색상 변경
    def change_ax_border_color(self, ax, color):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
        pass

    pass  # -- change_ax_border_color

    def plot_image( self, title="", cmap="gray", border_color="black"):
        # 그레이 스케일 이미지 표출
        global gs_row
        gs_row += 1
        gs_col = 1
        colspan = gs_col_cnt - gs_col

        ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

        img = self.img

        img_show = ax.imshow(img, cmap=cmap )

        ax.set_xlabel(title)
        ax.set_ylabel('y', rotation=0)

        border_color and self.change_ax_border_color(ax, border_color)

        fig.colorbar(img_show, ax=ax)

        return ax, img_show

    pass  # -- plot_image
    ''' -- 플롯 함수 '''

    def plot_histogram(self):  # 히스토 그램 표출

        img = self.img
        w, h = self.dimension()

        global gs_row
        gs_col = 0
        colspan = 1

        ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

        if not hasattr(self, "histogram") or self.histogram is None :
            self.make_histogram()
        pass

        histogram = self.histogram
        histogram_acc = self.histogram_acc

        max_y = 0

        if len( histogram ) > 10 :
            sum = histogram_acc[ - 1 ]
            f_10_sum = np.sum( histogram[ 0 : 10 ] )

            if f_10_sum > sum*0.8 :
                max_y = np.max( histogram[ 10 : ] )
            pass
        pass

        hist_avg = np.average(histogram)
        hist_std = np.std(histogram)
        hist_max = np.max(histogram)
        hist_med = np.median(histogram)

        log.info( f"hist avg = {hist_avg}, std = {hist_std}, med={hist_med}" )

        gs_avg = self.average()
        gs_max = self.max()
        gs_std = self.std()

        charts = {}

        if 1:
            # histogram bar chart
            y = histogram
            x = [ i for i, _ in enumerate(y)]

            width = 1 if len( histogram ) < 10 else 5

            charts["count"] = ax.bar(x, y, width=width, color='g', alpha=1.0)
        pass

        if 1:
            # accumulated histogram
            y = histogram_acc
            x = [i for i, _ in enumerate(y)]

            charts["accumulated"] = ax.plot(x, y, color='r', alpha=1.0)
        pass

        if 0 :
            # histogram std chart
            x = [gs_avg - gs_std, gs_avg + gs_std]
            y = [hist_max * 0.95, hist_max * 0.95]

            charts["std"] = ax.fill_between(x, y, color='cyan', alpha=0.5)
        pass

        if 0:
            # histogram average chart
            x = [ gs_avg ]
            y = [ hist_max ]
            charts["average"] = ax.bar(x, y, width=0.5, color='b', alpha=0.5)
        pass

        if 0:  # 레전드 표출
            t = []
            l = list(charts.keys())
            l = sorted(l)
            for k in l:
                t.append(charts[k])
            pass

            for i, s in enumerate(l):
                s = remove_space_except_first( s )
                l[i] = s[:4]
            pass

            loc = "upper right"

            if gs_avg > 122:
                loc = "upper left"
            pass

            ax.legend(t, l, loc=loc, shadow=True)
        pass  # -- 레전드 표출

        if 0:  # x 축 최대, 최소 설정
            max_x = gs_avg + gs_std * 1.2

            ax.set_xlim(0, max_x)
        pass

        if 1 :
            xlim_fr = 0
            xlim_to = len( histogram ) - 1
            xlim_to = xlim_to if xlim_to > 1 else 2.5 # data visualization
            xlim_fr = xlim_fr if xlim_to > 1 else -0.5  # data visualization

            ax.set_xlim( xlim_fr, xlim_to  )

            if not max_y :
                max_y = np.max( histogram )
            pass

            if max_y > 1_000 :
                ax.set_yscale('log')
            else :
                ax.set_ylim(0, max_y )
            pass
        pass

        histo_len = len(histogram)
        if histo_len > 10 :
            ax.set_xticks(np.arange(0, 250, 50))
        pass

        ax.grid( 1 )
        #x.set_ylabel('count', rotation=0)
        ax.set_xlabel( "Histogram")
    pass
    # -- plot_histogram

    # TODO  통계 함수

    def average(self):
        return np.average( self.img )
    pass

    def std(self):
        return np.std( self.img )
    pass

    def max(self):
        return np.max( self.img )
    pass

    ''' 통계 함수 '''

    # grayscale 변환 함수
    def convert_to_grayscale( self ) :
        log.info( "Convert to grayscale...." )

        img = self.img

        # TODO   채널 분리
        # b, g, r 채널 획득
        # cv2.imread() 는 b, g, r 순서대로 배열에서 반환한다.
        b_channel = img[:, :, 0].copy()
        g_channel = img[:, :, 1].copy()
        r_channel = img[:, :, 2].copy()

        # RGB -> GrayScale 변환 공식
        # average  Y = (R + G + B / 3)
        # weighted Y = (0.3 * R) + (0.59 * G) + (0.11 * B)
        # Colorimetric conversion Y = 0.2126R + 0.7152G  0.0722B
        # OpenCV CCIR Y = 0.299 R + 0.587 G + 0.114 B

        grayscale = r_channel*0.299 + g_channel*0.587 + b_channel*0.114

        grayscale = grayscale.astype( np.int16 )

        return Image( grayscale )
    pass
    # -- grayscale 변환

    def width(self):
        # image width
        img = self.img
        w = len( img [0])

        return w
    pass

    def height(self):
        # image height
        img = self.img
        h = len( img )

        return h
    pass

    def dimension(self):
        return self.width(), self.height()
    pass

    # TODO   영상 역전 함수
    def reverse_image( self, max=None):
        log.info("Reverse image....")

        img = self.img

        if max is None:
            max = np.max(img)

            if max < 1:
                max = 1
            elif max > 1:
                max = 255
            else:
                max = 1
            pass
        pass

        self.img = max - img

        return self
    pass
    # -- 영상 역전 함수

    # TODO   잡음 제거     # Median Blur Filter 적용

    # 잡음 제거 함수
    def remove_noise(self, ksize=3):
        msg = "Remove noise"
        log.info("%s ..." % msg)

        img = self.img

        algorithm = "cv.bilateralFilter"

        if "cv.bilateralFilter" == algorithm:
            log.info("cv.bilateralFilter(img,ksize,75,75)")

            img = img.astype(np.uint8)

            data = cv2.bilateralFilter(img, ksize, 75, 75)
        elif "cv2.medianBlur" == algorithm:
            log.info("cv2.medianBlur( image, ksize )")
            data = cv2.medianBlur(img, ksize)
        else:
            h = len(img)  # image height
            w = len(img[0])  # image width

            b = int(ksize / 2)

            data = np.empty([h, w], dtype=img.dtype)

            idx = 0
            for y in range(height):
                for x in range(width):
                    y0 = y - b
                    x0 = x - b

                    if y0 < 0:
                        y0 = 0
                    pass

                    if x0 < 0:
                        x0 = 0
                    pass

                    window = img[y0: y + b + 1, x0: x + b + 1]
                    median = np.median(window)
                    data[y][x] = median

                    0 and log.info( f"[{idx:05d}] data[{y}][{x}] = {median:.4f}" )
                    idx += 1
                pass
            pass
        pass

        log.info( f"Done. {msg}" )

        return Image( img=data, algorithm=algorithm)

    pass  # -- 잡음 제거 함수

    # TODO     Histogram 생성

    @profile
    def make_histogram(self):
        # this code is too slow
        msg = "Make histogram ..."
        log.info(msg)

        img = self.img

        size = 256 if np.max( img ) > 1 else 2

        histogram = [0] * size

        for row in img:
            for gs in row:
                histogram[gs] += 1
            pass
        pass

        histogram = np.array(histogram, 'u8')

        log.info( f"Done. {msg}" )

        histogram_acc = self.accumulate_histogram( histogram )

        self.histogram = histogram
        return histogram, histogram_acc

    pass  # -- get_histogram

    # TODO    누적 히스토 그램
    @profile
    def accumulate_histogram(self, histogram):
        msg = "Accumulate histogram ..."
        log.info(msg)

        histogram_acc = np.add.accumulate(histogram)

        self.histogram_acc = histogram_acc

        log.info( f"Done. {msg}" )

        return histogram_acc
    pass  # 누적 히스트 그램

    # TODO    히스토그램 평활화
    @profile
    def normalize_image_by_histogram(self):
        msg = "Normalize histogram"
        log.info( f"{msg} ..." )

        # https://en.wikipedia.org/wiki/Histogram_equalization

        img = self.img

        w, h = self.dimension()

        data = np.empty([h, w], dtype=img.dtype)

        # -histogram 생성
        _, histogram_acc = self.make_histogram()

        MN = h * w
        L = len(histogram_acc)

        cdf = histogram_acc

        # cdf_min = cdf[ 0 ]
        cdf_min = cdf[0]
        for c in cdf:
            if cdf_min == 0:
                cdf_min = c
            else:
                break
            pass
        pass

        log.info( f"cdf_min = {cdf_min}" )

        idx = 0
        L_over_MN_cdf_min = L/(MN - cdf_min + 0.0)

        cdf = np.array(cdf, 'float')

        cdf -= cdf_min
        cdf *= L_over_MN_cdf_min
        cdf += 0.5

        for y, row in enumerate(img):
            for x, gs in enumerate(row):
                # data[y][x] = int( (cdf[gs] - cdf_min)*L_over_MN_cdf_min + 0.5 )

                data[y][x] = int(cdf[gs])

                0 and log.info( f"[{idx:05d}] gs = {gs}, v={data[y][x]:0.4f}" )
                idx += 1
            pass
        pass

        log.info( f"Done. {msg}" )

        image = Image( data )

        return image
    pass
    # -- normalize_image_by_histogram

    ''' 이진화 '''
    # TODO     전역 임계치 처리
    def threshold_golobal(self, threshold=None):
        log.info("Global Threshold")

        reverse_required = 0

        img = self.img

        if not threshold:
            threshold = np.average(img)
        pass

        log.info( f"Threshold = {threshold}" )

        w, h = self.dimension()

        data = np.empty((h, w), dtype='B')

        for y, row in enumerate(img):
            for x, gs in enumerate(row):
                gs = round(gs)  # 반올림.
                data[y][x] = [0, 1][gs >= threshold]
            pass
        pass

        image = Image( data )
        image.threshold = threshold
        image.algorithm = "global thresholding"
        image.reverse_required = reverse_required

        return image
    pass  # -- 전역 임계치 처리

    # TODO     지역 평균 적응 임계치 처리
    def threshold_adaptive_mean(self, bsize=3, c=0):
        log.info("Apdative threshold mean")

        reverse_required = 1

        img = self.img

        w, h = self.dimension()

        data = np.empty((h, w), dtype='B')

        b = int(bsize / 2)
        if b < 1:
            b = 1
        pass

        for y, row in enumerate(img):
            for x, gs in enumerate(row):
                y0 = y - b
                x0 = x - b

                if y0 < 0:
                    y0 = 0
                pass

                if x0 < 0:
                    x0 = 0
                pass

                window = img[y0: y + b + 1, x0: x + b + 1]
                window_avg = np.average(window)
                threshold = window_avg - c

                data[y][x] = [0, 1][gs >= threshold]
            pass
        pass

        image = Image( data )
        image.threshold = -1
        image.algorithm = "adaptive mean thresholding"
        image.reverse_required = reverse_required

        return image
    pass  # -- 지역 평균 적응 임계치 처리

    # TODO     지역 가우시안 적응 임계치 처리
    def threshold_adaptive_gaussian(self, bsize=3, c=0):
        if 1:
            v = self.threshold_adaptive_gaussian_opencv(bsize=bsize, c=c)
        else:
            v = self.threshold_adaptive_gaussian_my(bsize=bsize, c=c)
        pass

        return v
    pass # -- threshold_adaptive_gaussian

    def threshold_adaptive_gaussian_opencv(self, bsize=3, c=0):
        msg = "Apdative threshold gaussian opencv"
        log.info(msg)
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

        reverse_required = 1
        bsize = 2 * int(bsize / 2) + 1

        img = self.img
        img = img.astype(np.uint8)

        data = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, c)

        image = Image( data )
        image.threshold = f"bsize = {bsize}"
        image.algorithm = "adaptive gaussian thresholding opencv"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_adaptive_gaussian_opencv

    def threshold_adaptive_gaussian_my(self, bsize=3, c=0):
        log.info("Apdative threshold gaussian my")

        # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel

        reverse_required = 1

        bsize = 2 * int(bsize / 2) + 1

        w, h = self.dimension()

        data = np.empty((h, w), dtype='B')

        b = int(bsize / 2)
        if b < 1:
            b = 1
        pass

        # the threshold value T(x,y) is a weighted sum (cross-correlation with a Gaussian window)
        # of the blockSize×blockSize neighborhood of (x,y) minus C

        img = self.img

        image_pad = np.pad(img, b, 'constant', constant_values=(0))

        def gaussian(x, y, bsize):
            #  The default sigma is used for the specified blockSize
            sigma = bsize
            # ksize	Aperture size. It should be odd ( ksizemod2=1 ) and positive.
            # sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
            ss = sigma * sigma
            pi_2_ss = 2 * math.pi * ss

            b = int(bsize / 2)

            x = x - b
            y = y - b

            v = math.exp(-(x * x + y * y) / ss) / pi_2_ss
            # g(x,y) = exp( -(x^2 + y^2)/s^2 )/(2pi*s^2)

            return v

        pass  # -- gaussian

        def gaussian_sum(window, bsize):
            gs_sum = 0

            # L = len( window )*len( window[0] )
            for y, row in enumerate(window):
                for x, v in enumerate(row):
                    gs_sum += v * gaussian(y, x, bsize)
                pass
            pass

            return gs_sum

        pass  # -- gaussian_sum

        for y, row in enumerate(image_pad):
            for x, gs in enumerate(row):
                if (b <= y < len(image_pad) - b) and (b <= x < len(row) - b):
                    window = image_pad[y - b: y + b + 1, x - b: x + b + 1]

                    threshold = gaussian_sum(window, bsize) - c

                    data[y - b][x - b] = [0, 1][gs >= threshold]
                pass
            pass
        pass

        return Image( data ), ("bsize = %s" % bsize), "adaptive gaussian thresholding my", reverse_required

    pass  # -- 지역 가우시안 적응 임계치 처리

    # TODO 이진화 계산
    def binarize_image(self, threshold=None):
        v = None

        if 1:
            w, h = self.dimension()
            bsize = w if w > h else h
            bsize = bsize / 2

            bsize = 5  # for line detection
            v = self.threshold_adaptive_gaussian(bsize=bsize, c=5)
        elif 0:
            bsize = 3
            v = self.threshold_adaptive_mean(bsize=bsize, c=0)
        else:
            threshold = np.average( self.img ) * 1.1
            v = self.threshold_golobal( threshold=threshold )
        pass

        return v
    pass # -- binarize_image
    ''' 이진화 계산 '''

    def count_y_axis_signal(self, ksize):
        msg = "y axis signal count"
        log.info(msg)

        img = self.img

        w, h = self.dimension()

        y_signal_counts = np.zeros( w, dtype='int')
        ksize = 1

        for x in range(w):
            window = img[0: h, x: x + ksize]
            signal_count = np.count_nonzero(window == 1)  # 검정 바탕 흰색 카운트
            # signal_count = np.count_nonzero( window == 0 ) # 흰 바탕 검정 카운트
            y_signal_counts[x] = signal_count
        pass

        log.info( f"Done. {msg}" )

        return y_signal_counts
    pass  # -- count_y_axis_signal

    def plot_y_counts(self, y_signal_counts, sentence ):
        # y count 표출
        ax, img_show = self.plot_image( title="y count" , cmap="gray", border_color="blue")
        self.plot_histogram()

        w, h = self.dimension()

        charts = { }

        if 1 :
            # y count bar chart
            y = y_signal_counts
            x = [i for i, _ in enumerate( y ) ]
            charts["y count"] = ax.bar(x, y, width=.6, color='y', align='center', alpha=1.0)
        pass

        if 1 :
            # word segments coordinate
            seginfos = self.word_seginfos( y_signal_counts, sentence )
            for seginfo in seginfos :
                x = seginfo.coord
                y = [ h ] * len( x )
                charts["word"] = ax.fill_between(x, y, color='r', alpha=0.5)
            pass
        pass

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

        ax.set_ylabel( 'Y count', rotation=90 )
        ax.set_xlim( 0, width )
    pass
    #-- y count 표출

    def save_excel_file(self, y_signal_counts):
        # 엑셀 파일 저장
        folder = "C:/Temp"

        global img_path

        file_name = file_name_except_path_ext( img_path )

        if 0 :
            # TODO     y count 데이터를 csv 파일로 저장
            path = f"{folder}/{file_name}_y_counts.csv"
            y_signal_counts.tofile( path, sep=',', format='%s')
            log.info(f"CSV file {path} was saved.")
        pass

        # TODO     y count 데이터를 엑셀 파일로 저장
        import xlsxwriter

        # Create a workbook and add a worksheet.
        excel_file_name = f"{folder}/{file_name}_y_counts.xlsx"
        workbook = xlsxwriter.Workbook( excel_file_name )
        worksheet = workbook.add_worksheet()

        row = 0

        # Iterate over the data and write it out row by row.
        velocity = np.diff( y_signal_counts )
        accel = np.diff( velocity )
        accel = accel/3.0 # data visualization


        cell_data_list = {}
        cell_data_list["y_count"]  = { "data" : y_signal_counts, 'type': 'column' , 'line': {'color': '#FF9900'}, }
        cell_data_list["velocity"] = { "data" : velocity       , 'type': 'line', 'line': {'color': 'blue'} }
        cell_data_list["accel"]    = { "data" : accel   , 'type': 'line', 'line': {'color': 'green'} }

        cell_data_key_list = [ "y_count", "velocity", "accel" ]

        for key in cell_data_key_list :
            col = 0
            worksheet.write( row, col, key )
            cell_data = cell_data_list[ key ]

            for cell_value in list( cell_data["data"] ) :
                col += 1
                worksheet.write(row, col, cell_value)
            pass

            row += 1
        pass

        if 1:  # 챠트 추가
            # https://xlsxwriter.readthedocs.io/chart.html

            chart = workbook.add_chart({'type': "line"})

            for i, key in enumerate( cell_data_key_list ):
                excel_row = i + 1

                cell_data = cell_data_list[key].copy()
                data = cell_data.pop( "data" )
                type = cell_data.pop( "type" )

                # Add a series to the chart.
                data_len = 1 + len( data )

                excel_letter = xlsxwriter.utility.xl_col_to_name( data_len )
                # excel_letter = to_excel_letter( col_number )

                # Add a series to the chart.
                cell_data_list_len = len(cell_data_list)
                chart.add_series({'categories': f'=Sheet1!A{excel_row}:A{excel_row}'})

                series_values = f"=Sheet1!B{excel_row}:{excel_letter:}{excel_row}"

                log.info( f"seires_values = {series_values}" )

                series = { 'values': series_values, "name" : key }

                series.update( cell_data )

                chart.add_series( series )
            pass

            # chart.set_size({'width': 720, 'height': 576})
            # Same as:
            # The default chart width x height is 480 x 288 pixels.
            chart.set_size({'x_scale': 3.5, 'y_scale': 2.4})

            chart.set_title({'name': 'Y signal count'})
            chart.set_x_axis({
                'name': 'x',
                'name_font': {'size': 14, 'bold': True},
                'num_font': {'italic': True},
            })

            # Insert the chart into the worksheet.
            worksheet.insert_chart(f'B{ len(cell_data_key_list) + 2}', chart)
        pass # 챠트 추가

        workbook.close()

        log.info( f"Excel file {excel_file_name} was saved." )

        # 탐색기 창을 뛰움.
        # 결과창 폴더 열기
        explorer_open(folder)

        # 결과창 엑셀 파일 열기
        show_excel_file = 0
        show_excel_file and explorer_open(excel_file_name)
    pass # --  #TODO y count 데이터를 엑셀, csv 파일로 저장

    def coords_of_word_segments_absolute(self, sentence, y_signal_counts):
        # 단어 짜르기
        # 정답에서 스페이스(" ")가 몇 개 들어가 있는 확인함.
        words_len = sentence.count(" ") + 1

        img = self.img

        h = len(img)
        w = len(img[0])

        # 단어 갯수 만큼 무식하게 일단 짜름.
        coords = []

        ref_y = np.average(y_signal_counts)
        ref_y = 1 / 5

        dw = int(w / words_len)
        x_0 = 0

        while x_0 < w:
            coords.append(x_0)
            x_0 += dw
        pass

        coords.append(w)

        return coords
    pass # coords_of_word_segments_absolute

    def word_seginfos(self, y_signal_counts, sentence ):
        img = self.img

        h = len(img)
        w = len(img[0])

        # 단어들 사이의 갭을 구한다.

        gaps = []

        ref_y = np.average( y_signal_counts )
        ref_y = ref_y/5.0 # gap 기준 높이

        prev_x = 0
        under_ref_running = False

        for x, y in enumerate( y_signal_counts ):
            is_under_ref = y < ref_y

            if x == 0 :
                if is_under_ref :
                    prev_x = 0
                pass
            elif under_ref_running :
                if not is_under_ref :
                    if prev_x is not None :
                        gaps.append( Gap( [prev_x, x] ) )
                        prev_x = None
                    pass
                pass
            elif not under_ref_running :
                if is_under_ref :
                    prev_x = x
                pass
            pass

            under_ref_running = is_under_ref
        pass

        if prev_x and prev_x < w - 1:
            x = w - 1
            gaps.append( Gap( [prev_x, x] ))
        pass

        def compare_gap( one, two ) :
            return two.distance() - one.distance()
        pass

        from functools import cmp_to_key
        gaps = sorted( gaps, key=cmp_to_key(compare_gap) )

        # 정답에서 스페이스(" ")가 몇 개 들어가 있는 확인함.
        words_len = sentence.count(" ") + 1

        gaps = gaps[ 0 : words_len ]

        seginfos = []
        prev_gap = None

        for curr_gap in gaps :
            if prev_gap is None :
                prev_gap = curr_gap
            else :
                coord = [ prev_gap.coord[1] , curr_gap.coord[0] ]
                seginfos.append( SegInfo( coord ) )
            pass
        pass

        return gaps
    pass # -- word_seginfos

    def word_segements(self, y_signal_counts, sentence ):
        # 단어 짜르기

        img = self.img

        h = len( img )

        # 단어 갯수 만큼 무식하게 일단 짜름.
        image_words = []

        seginfos = self.word_seginfos( y_signal_counts, sentence )

        for seginfo in seginfos :
            coord = seginfo.coord
            img_word = img[ 0 : h, coord[0] : coord[1] ]
            image_words.append( Image( img_word ) )
        pass

        return image_words

    pass # -- word_segements

pass
# -- class Image

image_org = Image( img_org )
image_org.save_img_as_file( "org" )
title = f'Original Image: { img_path.split("/")[-1] }'
0 and image_org.plot_image( title = title , cmap=None, border_color = "green" )

grayscale = image_org.convert_to_grayscale()
grayscale.reverse_image( max = 255 )
grayscale.save_img_as_file( "grayscale" )
grayscale.plot_image( title="Grayscale", cmap="gray", border_color = "green" )
grayscale.plot_histogram()

gs_avg = grayscale.average( )
gs_std = grayscale.std( )
sg_max = grayscale.max( )

log.info( f"grayscale avg = {gs_avg}, std = {gs_std}" )
#-- grayscale 변환

# 잡음 제거
ksize = 3
noise_removed = grayscale.remove_noise( ksize = ksize )
curr_image = noise_removed
noise_removed.save_img_as_file( f"noise_removed({curr_image.algorithm})" )

title = f"Noise removed ({curr_image.algorithm}, ksize={ksize})"
noise_removed.plot_image( title=title, cmap="gray", border_color = "blue" )
noise_removed.plot_histogram()

# 평활화
image_normalized = noise_removed.normalize_image_by_histogram()

print_prof_last()

image_normalized.save_img_as_file( "image_normalized" )
image_normalized.plot_image( title = "Normalization", cmap="gray", border_color = "green" )
image_normalized.plot_histogram()

#TODO 이진화
bin_image = image_normalized.binarize_image()
curr_image = bin_image
if bin_image.reverse_required :
    bin_image = bin_image.reverse_image()
pass

bin_image.save_img_as_file( f"image_binarized({curr_image.algorithm})" )

title = f"Binarization ({curr_image.algorithm}, {curr_image.threshold})"
bin_image.plot_image( title=title, cmap="gray", border_color = "blue" )
bin_image.plot_histogram()
#-- 이진화

#TODO   Y 축 데이터 히스토그램

sentence = "오늘 비교적 온화한 날씨가"

y_signal_counts = bin_image.count_y_axis_signal( ksize = 1 )

bin_image.plot_y_counts( y_signal_counts, sentence )

bin_image.save_excel_file( y_signal_counts )

word_segments = bin_image.word_segements( y_signal_counts, sentence )

save = 0
if save :
    # 세그먼테이션 파일 저장
    for idx, word_segment in enumerate( word_segments ) :
        word_segment.save_img_as_file( f"word_{idx:02d}" )
    pass
pass

log.info( "Plot show....." )

plt.show()

log.info( "Good bye!")

# end