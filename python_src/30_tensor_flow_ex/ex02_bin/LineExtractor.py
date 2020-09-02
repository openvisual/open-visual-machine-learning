# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from Common import *

# 이미지 클래스 임포트
from Image import *

class LineExtractor ( Common ):

    def __init__(self):
        Common.__init__( self )
    pass

    def my_line_extract( self, img_path, qtLineExtractor = None ) :

        # TODO    원천 이미지 획득
        # 이미지를 파일로 부터 RGB 색상으로 읽어들인다.

        img_org = cv2.imread(img_path, 1)

        # 이미지 높이, 넓이, 채널수 획득
        height = img_org.shape[0]
        width = img_org.shape[1]
        channel_cnt = img_org.shape[2]

        log.info(f"Image path: {img_path}")
        log.info(f"Image width: {width}, height: {height}, channel: {channel_cnt}")

        image_org = Image( img_org )
        image_org.save_img_as_file( img_path, "org" )
        title = f'Original Image: { img_path.split("/")[-1] }'
        0 and image_org.plot_image( title=title , cmap=None, border_color = "green", qtLineExtractor=qtLineExtractor )

        curr_image = image_org

        if 1 :
            # -- grayscale 변환
            grayscale = image_org.convert_to_grayscale()
            grayscale.reverse_image( max=255 )
            grayscale.save_img_as_file( img_path, "grayscale" )
            grayscale.plot_image( title="Grayscale", border_color = "green", qtLineExtractor=qtLineExtractor )
            grayscale.plot_histogram( qtLineExtractor=qtLineExtractor)

            curr_image = grayscale
        pass

        if 1 :
            # TODO 잡음 제거
            ksize = 5
            noise_removed = curr_image.remove_noise( algorithm="gaussian blur", ksize = ksize )
            curr_image = noise_removed

            curr_image.save_img_as_file( img_path, f"noise_removed({curr_image.algorithm})" )
            title = f"Noise removed ({curr_image.algorithm}, ksize={ksize})"
            curr_image.plot_image( title=title, border_color = "blue", qtLineExtractor=qtLineExtractor )
            curr_image.plot_histogram(qtLineExtractor=qtLineExtractor)
        pass

        if 1 :
            # TODO 평활화
            normalized = curr_image.normalize_image_by_histogram()

            curr_image = normalized

            curr_image.save_img_as_file( img_path, "image_normalized" )
            curr_image.plot_image( title="Normalization", border_color = "green", qtLineExtractor=qtLineExtractor )
            curr_image.plot_histogram(qtLineExtractor=qtLineExtractor)
        pass

        if 1:
            # TODO Gradient
            gradient = curr_image.gradient(ksize=7, kernel_type="cross")

            curr_image = gradient

            curr_image.save_img_as_file(img_path, curr_image.algorithm)
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue")
            curr_image.plot_histogram()
        pass  # -- gradient

        if 1 :
            #TODO 이진화
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
            curr_image.plot_image( title=title, border_color="blue" )
        pass #-- 이진화

        if 1 :
            # TODO morphology
            morphology = curr_image.morphology( is_open=0, bsize=7, iterations=3, kernel_type="cross" )

            curr_image = morphology

            curr_image.save_img_as_file( img_path, curr_image.algorithm )
            curr_image.plot_image( title=curr_image.algorithm, border_color="blue" )
            0 and curr_image.plot_histogram()
        pass # -- morphology

        if 1 :
            # 허프 라인 추출
            hough = curr_image.hough_lines(merge_lines=0)
            hough.save_img_as_file(img_path, hough.algorithm)

            hough = curr_image.hough_lines(merge_lines=1)
            hough.save_img_as_file(img_path, hough.algorithm)
            hough.plot_image(title=hough.algorithm, border_color="blue")
        pass

    pass
pass # -- LineExtractor

if __name__ == '__main__':
    lineExtractor = LineExtractor()

    lineExtractor.chdir_to_curr_file()

    img_path = "../data_yegan/ex_01/_1018877.JPG"
    #img_path = "../data_yegan/ex_01/_1018881.JPG"

    lineExtractor.my_line_extract( img_path = img_path )

    # 결과창 폴더 열기
    folder = "c:/temp"
    lineExtractor.open_file_or_folder(folder)

    log.info("Plot show.....")

    plt.show()

    log.info("Good bye!")

pass # -- main

# end