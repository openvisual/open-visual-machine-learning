# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os
from Common import *

# 이미지 클래스 임포트
from Image import *

class LineExtractor ( Common ):

    def __init__(self):
        Common.__init__( self )
    pass

    def my_line_extract(self, img_path, qtUi = None, mode="A", lineListPrev=None) :

        # TODO    원천 이미지 획득
        # 이미지를 파일로 부터 RGB 색상으로 읽어들인다.

        if 1 :
            pass
        elif os.path.exists( img_path ) and os.path.isdir( img_path ):
            log.info(f"ERROR: img_path={img_path} is a directory.")
            return -1
        else :
            log.info(f"ERROR: img_path={img_path} is invalid.")
            return -2
        pass

        prev_dir = os.getcwd()

        if 1 :
            fileName = img_path
            directory = os.path.dirname(fileName)
            fileBaseName = os.path.basename(fileName)

            if directory :
                log.info(f"Pwd 1: {os.getcwd()}")

                prev_dir = os.getcwd()

                os.chdir( directory )
                log.info(f"Pwd 2: {os.getcwd()}")
            pass

            img_path = f"./{fileBaseName}"

            log.info(f"dir = {directory}, fileBase={fileBaseName}")
        pass

        log.info(f"img_path to read = {img_path}")

        img_org = cv2.imread(img_path, 1)

        if prev_dir :
            os.chdir( prev_dir )
            log.info(f"Pwd 3: {os.getcwd()}")
        pass

        if img_org is None :
            log.info( f"ERROR: Failed to read the image file( {img_path} ).")

            return -3
        pass

        # 이미지 높이, 넓이, 채널수 획득
        height = img_org.shape[0]
        width = img_org.shape[1]
        channel_cnt = img_org.shape[2]

        log.info(f"Image width: {width}, height: {height}, channel: {channel_cnt}")

        image_org = Image( img_org )
        image_org.save_img_as_file( img_path, "org" )
        title = f'Original Image: { img_path.split("/")[-1] }'
        0 and image_org.plot_image(title=title, cmap=None, border_color = "green", qtUi=qtUi, mode=mode)

        curr_image = image_org

        if 1 : # -- grayscale 변환
            grayscale = image_org.convert_to_grayscale()
            grayscale.reverse_image( max=255 )
            grayscale.save_img_as_file( img_path, "grayscale" )
            grayscale.plot_image(title="Grayscale", border_color = "green", qtUi=qtUi, mode=mode)
            grayscale.plot_histogram(qtUi=qtUi, mode=mode)

            curr_image = grayscale
        pass

        if 1 : # TODO 잡음 제거
            ksize = 5
            noise_removed = curr_image.remove_noise( algorithm="gaussian blur", ksize = ksize )
            curr_image = noise_removed

            curr_image.save_img_as_file( img_path, f"noise_removed({curr_image.algorithm})" )
            title = f"Noise removed ({curr_image.algorithm}, ksize={ksize})"
            curr_image.plot_image(title=title, border_color = "blue", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass

        if 1 : # TODO 평활화
            normalized = curr_image.normalize_image_by_histogram()

            curr_image = normalized

            curr_image.save_img_as_file( img_path, "image_normalized" )
            curr_image.plot_image(title="Normalization", border_color = "green", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass

        if 1: # TODO Gradient
            gradient = curr_image.gradient(ksize=7, kernel_type="cross")

            curr_image = gradient

            curr_image.save_img_as_file(img_path, curr_image.algorithm)
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass  # -- gradient

        if 1 : #TODO 이진화
            #algorithm = "threshold_otsu"
            algorithm = "threshold_isodata"
            #algorithm = "threshold_balanced"
            #algorithm = "threshold_adaptive_gaussian"
            #algorithm = "threshold_adaptive_mean"
            #algorithm = "threshold_global"

            bin_image = curr_image.threshold(algorithm=algorithm)
            if bin_image.reverse_required :
                bin_image = bin_image.reverse_image()
            pass

            curr_image = bin_image

            curr_image.save_img_as_file( img_path, f"image_binarized({curr_image.algorithm})" )
            title = f"Binarization ({curr_image.algorithm})"
            curr_image.plot_image(title=title, border_color="blue", qtUi=qtUi, mode=mode)
        pass #-- 이진화

        if 1 : # TODO morphology
            morphology = curr_image.morphology( is_open=0, bsize=7, iterations=3, kernel_type="cross" )

            curr_image = morphology

            curr_image.save_img_as_file( img_path, curr_image.algorithm )
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
        pass # -- morphology

        lineList = None

        if 1 : # 허프 라인 추출
            lineList = curr_image.extract_lines( merge_lines=0, img_path=img_path )
            hough = curr_image.plot_lines( lineList )
            hough.save_img_as_file(img_path, hough.algorithm)

            lineList = curr_image.extract_lines( merge_lines=1, img_path=img_path)
            hough = curr_image.plot_lines( lineList )
            hough.save_img_as_file(img_path, hough.algorithm)
            hough.plot_image(title=hough.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
        pass

        lineList.mode = mode

        return lineList
    pass
pass # -- LineExtractor

if __name__ == '__main__':
    lineExtractor = LineExtractor()

    lineExtractor.chdir_to_curr_file()

    #img_path = "../data_yegan/ex_01/_1018877.JPG"
    img_path = "../data_yegan/ex_01/_1018881.JPG"
    #img_path = r"C:\test\AAA\_1018843.JPG"

    lineListA = lineExtractor.my_line_extract( img_path=img_path, qtUi=None )

    if 1 :
        # 결과창 폴더 열기
        folder = "c:/temp"
        lineExtractor.open_file_or_folder(folder)

        log.info("Plot show.....")

        plt.show()
    pass

    log.info("Good bye!")

pass # -- main

# end