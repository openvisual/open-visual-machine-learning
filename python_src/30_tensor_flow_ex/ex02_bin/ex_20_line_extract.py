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

# profile import
from profile import *
# utility import
from util import *

# 이미지 클래스 임포트
from Image import *

log.info( "Done Import.".center( 80, "*") )

def my_line_extract() :

    action = "line extract"
    Image.action = action
    # 이미지를 파일로 부터 RGB 색상으로 읽어들인다.
    # img_path = "../data_ocr/sample_01/messi5.png"
    # img_path = "../data_ocr/sample_01/hist_work_01.png"
    # img_path = "../data_ocr/sample_01/gosu_01.png"

    img_path = "../data_yegan/ex_01/_1018877.JPG"
    #img_path = "../data_yegan/ex_01/1-56.JPG"

    # TODO    원천 이미지 획득

    # read original image
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
    grayscale.plot_image( title="Grayscale", cmap="gray", border_color = "green" )
    grayscale.plot_histogram()
    #-- grayscale 변환

    if 1 :
        # TODO 잡음 제거
        ksize = 5
        noise_removed = grayscale.remove_noise( algorithm="gaussian blur", ksize = ksize )
        curr_image = noise_removed
        noise_removed.save_img_as_file( img_path, f"noise_removed({curr_image.algorithm})" )

        title = f"Noise removed ({curr_image.algorithm}, ksize={ksize})"
        noise_removed.plot_image( title=title, cmap="gray", border_color = "blue" )
        noise_removed.plot_histogram()

        curr_image = noise_removed
    pass

    if 1 :
        # TODO Gradient
        gradient = curr_image.gradient(ksize=5, kernel_type="cross")
        gradient.save_img_as_file(img_path, gradient.algorithm)
        gradient.plot_image(title=gradient.algorithm, cmap="gray", border_color="blue")
        gradient.plot_histogram()

        curr_image = gradient
    pass # -- gradient

    if 1 :
        # TODO 평활화
        normalized = curr_image.normalize_image_by_histogram()
        normalized.save_img_as_file( img_path, "image_normalized" )
        normalized.plot_image( title = "Normalization", cmap="gray", border_color = "green" )
        normalized.plot_histogram()

        curr_image = normalized
    pass

    if 0 :
        # 라플라시안
        laplacian = curr_image.laplacian(ksize=5)
        laplacian.save_img_as_file(img_path, laplacian.algorithm)
        laplacian.plot_image(title="Laplacian", cmap="gray", border_color="green")
        laplacian.plot_histogram()

        curr_image = laplacian

        # 평활화 2
        normalized2 = curr_image.normalize_image_by_histogram()
        normalized2.save_img_as_file(img_path, "laplacian_normalized")
        normalized2.plot_image(title="Laplacian Normalization", cmap="gray", border_color="green")
        normalized2.plot_histogram()

        curr_image = normalized2
    pass

    #TODO 이진화
    #algorithm = "threshold_otsu"
    #algorithm = "threshold_isodata"
    #algorithm = "threshold_balanced"
    #algorithm = "threshold_adaptive_gaussian"
    #algorithm = "threshold_adaptive_mean"
    algorithm = "threshold_golobal"

    bin_image = curr_image.threshold(algorithm=algorithm)
    curr_image = bin_image
    if bin_image.reverse_required :
        bin_image = bin_image.reverse_image()
    pass

    #bin_image.reverse_image()

    bin_image.save_img_as_file( img_path, f"image_binarized({curr_image.algorithm})" )
    title = f"Binarization ({curr_image.algorithm})"
    bin_image.plot_image( title=title, cmap="gray", border_color = "blue" )
    bin_image.plot_histogram()
    #-- 이진화

    if 1 :
        # TODO morphology
        morphology = bin_image.morphology( is_open = 0, bsize = 3, iterations = 1, kernel_type="cross" )
        morphology.save_img_as_file( img_path, morphology.algorithm )
        morphology.plot_image( title=morphology.algorithm, cmap="gray", border_color = "blue" )
        morphology.plot_histogram()

        bin_image = morphology
    pass # -- morphology

    if 1 :
        # TODO Gradient 2
        gradient = bin_image.gradient(ksize=5, kernel_type="cross")
        gradient.save_img_as_file(img_path, gradient.algorithm)
        gradient.plot_image(title=gradient.algorithm, cmap="gray", border_color="blue")
        gradient.plot_histogram()

        bin_image = gradient
    pass # -- gradient 2

    log.info( "Plot show....." )

    print_prof_last()

    plt.show()

    log.info( "Good bye!")
pass

if __name__ == '__main__':
    # 현재 파일의 폴더로 실행 폴더를 이동함.
    chdir_to_curr_file()

    my_line_extract()
pass # -- main

# end