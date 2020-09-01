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
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from util import *

# 이미지 클래스 임포트
from Image import *

log.info( "Done Import.".center( 80, "*") )

def my_line_extract( img_path ) :

    action = "line extract"
    Image.action = action

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
    0 and image_org.plot_image( title=title , cmap=None, border_color = "green" )

    grayscale = image_org.convert_to_grayscale()
    grayscale.reverse_image( max=255 )
    grayscale.save_img_as_file( img_path, "grayscale" )
    grayscale.plot_image( title="Grayscale", border_color = "green" )
    grayscale.plot_histogram()
    #-- grayscale 변환

    if 1 :
        # TODO 잡음 제거
        ksize = 5
        noise_removed = grayscale.remove_noise( algorithm="gaussian blur", ksize = ksize )
        curr_image = noise_removed
        noise_removed.save_img_as_file( img_path, f"noise_removed({curr_image.algorithm})" )

        title = f"Noise removed ({curr_image.algorithm}, ksize={ksize})"
        noise_removed.plot_image( title=title, border_color = "blue" )
        noise_removed.plot_histogram()

        curr_image = noise_removed
    pass

    if 1 :
        # TODO 평활화
        normalized = curr_image.normalize_image_by_histogram()
        normalized.save_img_as_file( img_path, "image_normalized" )
        normalized.plot_image( title="Normalization", border_color = "green" )
        normalized.plot_histogram()

        curr_image = normalized
    pass

    if 1:
        # TODO Gradient
        gradient = curr_image.gradient(ksize=7, kernel_type="cross")
        gradient.save_img_as_file(img_path, gradient.algorithm)
        gradient.plot_image(title=gradient.algorithm, border_color="blue")
        gradient.plot_histogram()

        curr_image = gradient
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

        bin_image.save_img_as_file( img_path, f"image_binarized({bin_image.algorithm})" )
        title = f"Binarization ({bin_image.algorithm})"
        bin_image.plot_image( title=title, border_color="blue" )

        curr_image = bin_image
    pass #-- 이진화

    if 1 :
        # TODO morphology
        morphology = curr_image.morphology( is_open=0, bsize=7, iterations=3, kernel_type="cross" )
        morphology.save_img_as_file( img_path, morphology.algorithm )
        morphology.plot_image( title=morphology.algorithm, border_color="blue" )
        0 and morphology.plot_histogram()

        curr_image = morphology
    pass # -- morphology

    if 1 :
        # 허프 라인 추출
        hough = curr_image.hough_lines(merge_lines=0)
        hough.save_img_as_file(img_path, hough.algorithm)

        hough = curr_image.hough_lines(merge_lines=1)
        hough.save_img_as_file(img_path, hough.algorithm)
        hough.plot_image(title=hough.algorithm, border_color="blue")

        curr_image = hough
    pass


pass

if __name__ == '__main__':
    # 현재 파일의 폴더로 실행 폴더를 이동함.
    chdir_to_curr_file()

    img_path = "../data_yegan/ex_01/_1018877.JPG"
    #img_path = "../data_yegan/ex_01/_1018881.JPG"

    my_line_extract( img_path = img_path )

    print_prof_last()

    # 결과창 폴더 열기
    folder = "c:/temp"
    open_file_or_folder(folder)

    log.info("Plot show.....")

    plt.show()

    log.info("Good bye!")

pass # -- main

# end