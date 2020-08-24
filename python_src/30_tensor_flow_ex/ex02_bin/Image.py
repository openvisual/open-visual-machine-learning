# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

# profile import
from profile import *
# utility import
from util import *

import os, cv2, numpy as np, sys, time
import cv2 as cv
import math

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

    # 이미지 저장 회수
    img_save_cnt = 0
    clear_work_files = 0

    gs_row_cnt = 4
    gs_col_cnt = 7

    gs_row = -1
    gs_col = 0

    gridSpec = None
    plt_windows_cnt = 0

    def __init__(self, img, algorithm="" ):
        # 2차원 배열 데이터
        self.img = img
        self.algorithm = algorithm
        self.histogram = None
        self.histogram_acc = None
    pass

    def img_file_name(self, img_path, work):
        # C:/temp 폴더에 결과 파일을 저정합니다.

        folder = "C:/temp"

        img_save_cnt = Image.img_save_cnt

        fn = img_path

        root = fn[: fn.rfind("/")]

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

        fn_hdr = folder + fn[: k]

        if Image.clear_work_files and img_save_cnt == 0 :
            # fn_hdr 로 시작하는 모든 파일을 삭제함.
            import glob
            for f in glob.glob( f"{fn_hdr}*" ):
                log.info( f"file to delete {f}")
                try :
                    os.remove(f)
                except Exception as e:
                    error = str( e )
                    log.info( f"cannot file to delete. error: {error}" )
                pass
            pass
        pass

        fn = fn_hdr + ("_%02d_" % img_save_cnt) + work + fn[k:].lower()

        Image.img_save_cnt += 1

        return fn
    pass  # -- img_file_name

    def save_img_as_file(self, img_path, work, ):
        filename = self.img_file_name( img_path, work)
        img = self.img

        cmap = "gray"

        if len( img ) == 3 :
            cmap = None
        pass

        plt.imsave(filename, img, cmap=cmap)

        log.info( f"Image saved as file name[ {filename} ]" )
    pass  # -- save_img_as_file

    ''' -- 이미지 저장 함수 '''

    # pyplot ax 의 프레임 경계 색상 변경
    def change_ax_border_color(self, ax, color):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
        pass

    pass  # -- change_ax_border_color

    def plot_image( self, title="", border_color="black"):
        # TODO 이미지 그리기

        Image.gs_row += 1

        if Image.gridSpec is None or Image.gs_row >= Image.gs_row_cnt :
            Image.gs_row = 0

            Image.fig = plt.figure(figsize=(13, 10), constrained_layout=True)
            Image.gridSpec = GridSpec(Image.gs_row_cnt, Image.gs_col_cnt, figure=Image.fig)

            Image.plt_windows_cnt += 1

            cnt = Image.plt_windows_cnt

            # plot 윈도우 제목 변경
            win_title = Image.action.capitalize()
            plt.get_current_fig_manager().canvas.set_window_title( f"{win_title} {cnt}" )
        pass

        gs_col = 1
        colspan = Image.gs_col_cnt - gs_col

        ax = plt.subplot(Image.gridSpec.new_subplotspec((Image.gs_row, gs_col), colspan=colspan))

        img = self.img

        cmap = "gray"

        if len( img ) == 3 :
            cmap = None
        pass

        img_show = ax.imshow(img, cmap=cmap )

        ax.set_xlabel(title)
        ax.set_ylabel('y', rotation=0)

        border_color and self.change_ax_border_color(ax, border_color)

        Image.fig.colorbar(img_show, ax=ax)

        return ax, img_show

    pass  # -- plot_image
    ''' -- 플롯 함수 '''

    def plot_histogram(self):  # 히스토 그램 표출
        img = self.img
        h = len( img )
        w = len( img[0] )

        gs_col = 0
        colspan = 1

        ax = plt.subplot(Image.gridSpec.new_subplotspec((Image.gs_row, gs_col), colspan=colspan))

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

        if len( histogram ) > 10 :
            # histogram bar chart
            y = histogram
            x = range(len(y))

            width = 2

            import matplotlib.colors as mcolors

            clist = [(0, "blue"), (0.125, "green"), (0.25, "yellow"), (0.5, "cyan"), (0.7, "orange"), (0.9, "red"), (1, "blue")]
            rvb = mcolors.LinearSegmentedColormap.from_list("", clist)

            clist_ratio = len( clist )/np.max( y )

            #charts["count"] = ax.bar(x, y, width=width, color=rvb(y*clist_ratio ) )

            charts["count"] = ax.bar(x, y, width=width, color='g', alpha=1.0)
        else :
            y = histogram
            x = range(len(y))
            width = 1

            charts["count"] = ax.bar(x, y, width=width, color='g', alpha=1.0)
        pass

        if 1:
            # accumulated histogram
            y = histogram_acc
            x = range( len(y) )

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

        if 1 :
            xlim_fr = 0
            xlim_to = len( histogram )
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
            ax.set_xticks( [ 0, 50, 100, 150, 200, 255 ] )
        pass

        ax.grid( 1 )
        #x.set_ylabel('count', rotation=0)
        ax.set_xlabel( "Histogram")
    pass
    # -- plot_histogram

    # TODO  통계 함수

    def average(self): # 평균
        return np.average( self.img )
    pass

    def std(self): # 표준 편차
        return np.std( self.img )
    pass

    def max(self): # 최대값
        return np.max( self.img )
    pass

    # -- 통계 함수

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
    pass # -- convert_to_grayscale

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

    def dimension_ratio(self):
        return self.width()/self.height()
    pass

    def reverse_image( self, max=None):
        # TODO   영상 역전 함수

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
    pass # -- reverse_image

    def laplacian(self, ksize=5):
        # TODO   라플라시안
        # https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html

        msg = "Laplacian"
        log.info( f"{msg} ..." )

        ksize = 2*int(ksize/2) + 1

        img = self.img

        algorithm = f"laplacian ksize={ksize}"

        img = img.astype(np.float)

        data = cv.Laplacian(img, cv.CV_64F)

        # normalize to gray scale
        min = np.min( data )
        max = np.max( data )

        data = (255/(max - min))*(data - min)

        min = np.min(data)
        max = np.max(data)
        # -- # normalize to gray scale

        data = data.astype(np.int)

        min = np.min(data)
        max = np.max(data)

        log.info( f"min = {min}, max={max}")

        log.info( f"Done. {msg}" )

        return Image( img=data, algorithm=algorithm)
    pass  # -- laplacian

    def gradient(self, ksize=5, kernel_type="cross"):
        # TODO   그라디언트
        # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html

        msg = "Gradient"
        log.info( f"{msg} ..." )

        ksize = 2*int(ksize/2) + 1

        img = self.img

        algorithm = f"Gradient ksize={ksize}, ktype={kernel_type}"

        img = img.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

        if kernel_type == "rect":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        elif kernel_type == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
        elif kernel_type == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        pass

        data = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

        log.info( f"Done. {msg}" )

        return Image( img=data, algorithm=algorithm)
    pass  # -- gradient

    def remove_noise(self, algorithm , ksize=5 ):
        # TODO   잡음 제거
        msg = "Remove noise"
        log.info( f"{msg} ..." )

        img = self.img

        if algorithm == "gaussian blur"  :
            # Gaussian filtering
            algorithm = f"{algorithm} ksize={ksize}"

            img = img.astype(np.uint8)
            data = cv.GaussianBlur(img, (ksize, ksize), 0)
        elif algorithm == "bilateralFilter" :
            algorithm = f"{algorithm} ksize={ksize}, 75, 75"

            img = img.astype(np.uint8)
            data = cv2.bilateralFilter(img, ksize, 75, 75)
        elif algorithm == "medianBlur" :
            algorithm = f"{algorithm} ksize={ksize}"

            data = cv2.medianBlur(img, ksize)
        pass

        log.info( f"Done. {msg}" )

        return Image( img=data, algorithm=algorithm)
    pass  # -- remove_noise

    @profile
    def make_histogram(self):
        # TODO     Histogram 생성
        msg = "Make histogram ..."
        log.info(msg)

        img = self.img

        min = np.min( img )
        max = np.max( img )

        log.info( f"hist img min={min}, max={max}")

        size = 256 if np.max( img ) > 1 else 2

        histogram = [0] * size

        for row in img:
            for gs in row:
                gs = int( gs )
                histogram[gs] += 1
            pass
        pass

        histogram = np.array( histogram )

        log.info( f"Done. {msg}" )

        histogram_acc = self.accumulate_histogram( histogram )

        self.histogram = histogram

        return histogram, histogram_acc
    pass  # -- make_histogram

    @profile
    def accumulate_histogram(self, histogram):
        # TODO    누적 히스토 그램

        msg = "Accumulate histogram ..."
        log.info(msg)

        histogram_acc = np.add.accumulate(histogram)

        self.histogram_acc = histogram_acc

        log.info( f"Done. {msg}" )

        return histogram_acc
    pass  # accumulate_histogram

    @profile
    def normalize_image_by_histogram(self):
        # TODO    히스토그램 평활화

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

        cdf = np.array(cdf, 'float')

        cdf = (cdf - cdf_min)*(L-1)/(MN - cdf_min)

        for y, row in enumerate(img):
            for x, gs in enumerate(row):
                gs = int( gs )

                data[y][x] = int(cdf[gs])

                0 and log.info( f"[{idx:05d}] gs = {gs}, v={data[y][x]:0.4f}" )
                idx += 1
            pass
        pass

        log.info( f"Done. {msg}" )

        image = Image( data )

        return image
    pass # -- normalize_image_by_histogram

    ''' 이진화 '''
    # TODO     전역 임계치 처리
    def threshold_golobal(self ):
        msg = "Threshold global"
        log.info( f"{msg}" )

        reverse_required = 0

        img = self.img

        h = len( img )
        w = len( img[0] )

        histogram = None

        if not hasattr(self, "histogram" ) or self.histogram is None :
            histogram , _ = self.make_histogram()
        else :
            histogram = self.histogram
        pass

        histogram = histogram.copy()

        margin = 20
        histogram[0: margin] = 0

        x = np.arange(0, len(histogram))

        avg =  margin + ( sum( histogram * x )/np.sum( histogram ) )

        threshold = avg

        data = np.where( img >= threshold , 1, 0 )

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"global thresholding ({ int(threshold) })"
        image.reverse_required = reverse_required

        return image
    pass  # -- 전역 임계치 처리

    def threshold_isodata(self ):
        msg = "Threshold isodata"
        log.info( f"{msg}")

        reverse_required = 0

        img = self.img

        histogram, _ = self.make_histogram()

        threshold = 0
        t_diff = None
        for i in range(2, 256):
            mL_hist = histogram[0: i]
            mL = sum(mL_hist * np.arange(0, i)) / sum(mL_hist)
            mH_hist = histogram[i:]
            mH = sum(mH_hist * np.arange(i, 256)) / sum(mH_hist)

            diff = abs(i - (mL + mH) / 2)
            if t_diff is None or diff < t_diff:
                t_diff = diff
                threshold = i
            pass
        pass

        log.info( f"Threshold isodata = {threshold}" )

        data = np.where( img >= threshold , 1, 0 )

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"threshold isodata ({ int(threshold) })"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_isodata

    def threshold_balanced(self ):
        msg = "Threshold balanced"
        log.info( f"{msg}")

        reverse_required = 0

        img = self.img

        histogram, _ = self.make_histogram()

        threshold = 0
        t_diff = None
        for i in range(2, 256):
            mL_hist = histogram[0: i]
            mH_hist = histogram[i:]

            diff = abs(sum(mL_hist) - sum(mH_hist))
            if t_diff is None or diff < t_diff:
                t_diff = diff
                threshold = i
            pass
        pass

        log.info( f"Threshold balanced = {threshold}" )

        data = np.where( img >= threshold , 1, 0 )

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"threshold balanced ({ int(threshold) })"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_balanced

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
        image.algorithm = "adaptive mean"
        image.reverse_required = reverse_required

        return image
    pass  # -- 지역 평균 적응 임계치 처리

    def threshold_otsu(self):
        msg = "threshold otsu"
        log.info(msg)
        # https: // docs.opencv.org / 3.4 / d7 / d4d / tutorial_py_thresholding.html

        reverse_required = 0

        img = self.img
        img = img.astype(np.uint8)

        # Gaussian filtering
        #blur = cv.GaussianBlur(img, (5, 5), 0)
        # Otsu's thresholding
        threshold, data = cv.threshold( img, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"otsu threshold={threshold}"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_otsu_opencv

    # TODO     지역 가우시안 적응 임계치 처리
    def threshold_adaptive_gaussian(self, bsize=5, c=0):
        algorithm = 0

        if algorithm == 0 :
            v = self.threshold_adaptive_gaussian_opencv(bsize=bsize, c=c)
        elif algorithm == 1:
            v = self.threshold_adaptive_gaussian_my(bsize=bsize, c=c)
        pass

        return v
    pass # -- threshold_adaptive_gaussian

    def threshold_adaptive_gaussian_opencv(self, bsize=5, c=0):
        msg = "Apdative threshold gaussian opencv"
        log.info(msg)
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

        reverse_required = 0
        bsize = 2 * int(bsize / 2) + 1

        img = self.img
        img = img.astype(np.uint8)

        data = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, c)

        image = Image( data )
        image.threshold = f"bsize = {bsize}"
        image.algorithm = f"adaptive gaussian, bsize={bsize}"
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
            #sigma = bsize
            # ksize	Aperture size. It should be odd ( ksizemod2=1 ) and positive.
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
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

        bsize_square = bsize*bsize

        for y, row in enumerate(image_pad):
            for x, gs in enumerate(row):
                if (b <= y < len(image_pad) - b) and (b <= x < len(row) - b):
                    window = image_pad[y - b: y + b + 1, x - b: x + b + 1]

                    gaussian_avg = gaussian_sum(window, bsize)/bsize_square

                    threshold = gaussian_avg - c

                    data[y - b][x - b] = [0, 1][gs >= threshold]
                pass
            pass
        pass

        image = Image(data)
        image.threshold = f"bsize = {bsize}"
        image.algorithm = "adaptive gaussian thresholding my"
        image.reverse_required = reverse_required

        return image
    pass  # -- 지역 가우시안 적응 임계치 처리

    def threshold(self, algorithm):
        # TODO 이진화

        v = None

        if "otsu" in algorithm :
            v = self.threshold_otsu()
        elif "gaussian" in algorithm :
            w, h = self.dimension()

            bsize = w if w > h else h
            bsize = bsize / 6
            bsize = 13

            v = self.threshold_adaptive_gaussian(bsize=bsize, c=0)
        elif "mean" in algorithm :
            bsize = 5
            v = self.threshold_adaptive_mean(bsize=bsize, c=0)
        elif "global" in algorithm :
            v = self.threshold_golobal()
        elif "isodata" in algorithm :
            v = self.threshold_isodata()
        elif "balanced" in algorithm:
            v = self.threshold_balanced()
        pass

        return v
    pass # -- binarize_image

    def morphology(self, is_open, bsize = 5, iterations = 1, kernel_type = "cross" ):
        msg = "morphology"
        log.info(msg)

        bsize = 2*int( bsize/2 ) + 1

        img = self.img
        img = img.astype(np.uint8)

        data = img

        if iterations < 1 :
            iterations = 1
        pass

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bsize, bsize))

        if kernel_type == "rect" :
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bsize, bsize))
        elif kernel_type == "cross" :
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (bsize, bsize))
        elif kernel_type == "ellipse" :
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bsize, bsize))
        pass

        for _ in range ( iterations ) :
            if is_open :
                data = cv2.erode( data, kernel, iterations = 1)
            else :
                data = cv2.dilate(data, kernel, iterations=1)
            pass

            if is_open :
                data = cv2.dilate( data, kernel, iterations=1)
            else :
                data = cv2.erode(data, kernel, iterations=1)
            pass
        pass

        op_close = "open" if is_open else "close"

        image = Image(data)
        image.algorithm = f"morphology, {op_close}, kernel={kernel_type}, bsize={bsize}, iterations={iterations}"

        return image
    pass  # -- morphology_closing

    def get_vertical_histogram(self, ksize):
        msg = "vertical_histogram"
        log.info(msg)
        # https://medium.com/@susmithreddyvedere/segmentation-in-ocr-10de176cf373

        img = self.img

        vertical_hist = np.sum(img,axis=0,keepdims=True)

        vertical_hist = vertical_hist[ 0 ]

        log.info( f"Done. {msg}" )

        return vertical_hist
    pass  # -- count_y_axis_signal

    def get_vertical_histogram_old(self, ksize):
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

    def plot_vertical_histogram(self, vertical_histogram, sentence):
        # y count 표출
        # word segments coordinate
        seginfos, gaps, ref_y, ref_ratio = self.word_seginfos(vertical_histogram, sentence)

        title = f"Vertical histogram( ref ratio={ ref_ratio*100:.2f}% )"

        ax, img_show = self.plot_image( title=title, border_color="blue")
        self.plot_histogram()

        w, h = self.dimension()

        charts = { }

        if 1 :
            # word segments coordinate
            if 1 :
                x = [ 0 , w ]
                y = [ ref_y, ref_y ]
                charts["ref_y"] = ax.plot(x, y, color='red', alpha=1.0, linestyle='solid')
            pass

            for gap in gaps :
                x = gap.coord
                y = [h] * len(x)
                charts["gaps"] = ax.fill_between(x, y, color='blue', alpha=0.6)
            pass

            for seginfo in seginfos :
                x = seginfo.coord
                y = [ h ] * len( x )
                charts["segments"] = ax.fill_between(x, y, color='r', alpha=0.4)
            pass
        pass

        if 1:
            # vertical histogram
            vertical_histogram = vertical_histogram.astype( np.int )

            y = vertical_histogram
            x = range(len(y))
            charts["y count"] = ax.bar(x, y, width=.6, color='yellow', align='center', alpha=1.0)

            from sklearn.preprocessing import MinMaxScaler
            velocity = np.diff(vertical_histogram)
            accel = np.diff(velocity)

            y = accel
            x = range(len(y))

            charts["accel"] = ax.plot(x, y, color='green', alpha=1.0, label="accel")
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
        ax.set_xlim( 0, w )
    pass
    #-- y count 표출

    def save_vertical_hist_as_excel(self, img_path, vertical_histogram, show_excel_file = 0):
        # 엑셀 파일 저장
        folder = "C:/Temp"

        file_name = file_name_except_path_ext( img_path )

        # TODO     y count 데이터를 엑셀 파일로 저장
        import xlsxwriter

        # Create a workbook and add a worksheet.
        excel_file_name = f"{folder}/{file_name}_y_counts.xlsx"

        # 쓰기 가능 여부 체크
        if not is_writable( excel_file_name ) :
            import datetime
            now = datetime.datetime.now()
            now_str = now.strftime('%m-%d %H%M%S')
            now_str = now_str.split( "." )[0]
            excel_file_name = f"{folder}/{file_name}_y_counts_{now_str}.xlsx"
        pass

        workbook = xlsxwriter.Workbook( excel_file_name )
        worksheet = workbook.add_worksheet()

        row = 0

        # Iterate over the data and write it out row by row.
        vertical_histogram = vertical_histogram.astype( np.int )
        velocity = np.diff(vertical_histogram)
        accel = np.diff( velocity )

        cell_data_list = {}
        cell_data_list["y count"]  = { "data" : vertical_histogram, 'type': 'column' , 'line': {'color': '#FF9900'}, }
        cell_data_list["velocity"] = { "data" : velocity       , 'type': 'line', 'line': {'color': 'blue'} }
        cell_data_list["accel"]    = { "data" : accel   , 'type': 'line', 'line': {'color': 'green'} }

        cell_data_key_list = [ "y count", "velocity", "accel" ]

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

            #chart_type = "line" #"bar" #"column"
            chart_type = "column"
            chart = workbook.add_chart({'type': chart_type }) # line

            for i, key in enumerate( cell_data_key_list ):
                excel_row = i + 1

                cell_data = cell_data_list[key].copy()
                data = cell_data.pop( "data" )
                type = cell_data.pop( "type" )

                # Add a series to the chart.
                data_len = 1 + len( data )

                excel_letter = xlsxwriter.utility.xl_col_to_name( data_len )

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
            chart.set_size({'x_scale': 2.5, 'y_scale': 1.8 })

            chart.set_title({'name': 'Vertical Histogram'})
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

        # 결과창 폴더 열기
        open_file_or_folder(folder)

        # 결과창 엑셀 파일 열기
        if show_excel_file :  open_file_or_folder(excel_file_name)
    pass # -- y count 데이터를 엑셀, csv 파일로 저장

    def word_seginfos(self, y_signal_counts, sentence ):
        # 세그먼트 정보 계산
        img = self.img

        h = len(img)
        w = len(img[0])

        # 단어들 사이의 갭을 구한다.

        gaps = []

        # 기준값 비율
        #ref_ratio = 1/3.0
        #ref_ratio = 1/3.5
        ref_ratio = 1/4.0

        # 평균치의 몇 프로를 기준값으로 설정.
        avg = np.average(y_signal_counts)
        ref_y = avg * ref_ratio

        prev_x = 0
        running_under_ref = False

        for x, y in enumerate( y_signal_counts ):
            is_under_ref = y < ref_y

            if x == 0 :
                if is_under_ref :
                    prev_x = 0
                pass
            elif running_under_ref :
                if not is_under_ref :
                    if prev_x is not None :
                        gaps.append( Gap( [prev_x, x] ) )
                        prev_x = None
                    pass
                pass
            elif not running_under_ref :
                if is_under_ref :
                    prev_x = x
                pass
            pass

            running_under_ref = is_under_ref
        pass

        if prev_x and prev_x < w - 1:
            x = w - 1
            gaps.append( Gap( [prev_x, x] ))
        pass

        for idx, gap in enumerate( gaps ) :
            gap.idx = idx
        pass

        gap_last = None

        if gaps :
            gap_last = gaps[ -1 ]
        pass

        def compare_gap_dist( one, two ) :
            return two.distance() - one.distance()
        pass

        from functools import cmp_to_key
        gaps = sorted( gaps, key=cmp_to_key(compare_gap_dist) )

        # 정답에서 스페이스(" ")가 몇 개 들어가 있는 확인함.
        words_count = sentence.count(" ") + 1

        gaps = gaps[ 0 : words_count ]

        def compare_gap_idx( one, two ) :
            return one.idx - two.idx
        pass

        from functools import cmp_to_key
        gaps = sorted( gaps, key=cmp_to_key(compare_gap_idx) )

        if gap_last and gaps[ -1 ].idx < gap_last.idx :
            gaps.append( gap_last )
        pass

        seginfos = []
        prev_gap = None

        for curr_gap in gaps :
            if prev_gap :
                coord = [ prev_gap.coord[1] , curr_gap.coord[0] ]
                seginfos.append( SegInfo( coord ) )
            pass

            prev_gap = curr_gap
        pass

        log.info( f"gap count = {len(gaps)}, seg count = {len(seginfos)}")

        return seginfos, gaps, ref_y, ref_ratio
    pass # -- word_seginfos

    def word_segements(self, y_signal_counts, sentence ):
        # 단어 짜르기
        img = self.img
        h = len( img )

        image_words = []

        seginfos, gaps , ref_y, ref_ratio = self.word_seginfos( y_signal_counts, sentence )

        for seginfo in seginfos :
            coord = seginfo.coord
            img_word = img[ 0 : h, coord[0] : coord[1] ]
            image_words.append( Image( img_word ) )
        pass

        return image_words
    pass # -- word_segements

    def hough_lines(self):
        # hough line 추출
        msg = "hough line"
        log.info( f"{msg}")

        img = self.img

        img = img.astype(np.uint8)

        if np.max( img ) < 2 :
            img = img*255
        pass

        lines = cv.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)

        data = cv.cvtColor( img, cv.COLOR_GRAY2BGR )

        import matplotlib.colors as mcolors
        from matplotlib.colors import hex2color, rgb2hex

        mcolors = mcolors.BASE_COLORS
        mcolors = mcolors.TABLEAU_COLORS

        colors = []

        for key, hex_color in enumerate( mcolors ) :
            color = hex2color(hex_color)
            color = tuple([int(255 * x) for x in color])

            colors.append( color )
            log.info(f"{name} = {hex_color} = {color}")
        pass

        colors_len = len(colors)

        for i, line in enumerate( lines ) :
            l = line[0]
            #color = (0, 0, 255)
            color = colors[ i%colors_len ]
            cv.line(data, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)
        pass

        image = Image(data)
        image.algorithm = "hough lines"

        log.info(f"Done. {msg}")

        return image
    pass # hough_lines

pass
# -- class Image

# end