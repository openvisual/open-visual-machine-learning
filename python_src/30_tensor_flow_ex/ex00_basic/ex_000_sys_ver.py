# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings 
warnings.filterwarnings('ignore',category=FutureWarning)

import sys

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# 시스템 버전을 나타내는 예제 입니다.

print( "" )
print( "" )
print( " Hello .... ".center( 80, "*" ) )
print( "" )

print( "Python version : %s" % sys.version )

print( "TensorFlow version : %s" % tf.__version__)

print( "Keras version : %s" % keras.__version__ )

try :
    import torch
    print( "Torch version : %s" % torch.__version__ )
    import torchvision
    print( "Torchvision version : %s" % torchvision.__version__ )
except :
    print( "Troch is not installed." )

try:
    # % opencv version
    import cv2
    print( "OpenCV version : %s" % cv2.__version__ )
except Exception as e :
    print( e )
    print( "OpenCV is not installed on this machine.")
pass

# print gpu spec
if 0 : 
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
pass

if 0 : 
    import tensorflow as tf
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    pass
pass
#-- print gpu spec

print( "" )
print( " Good bye! ".center( 80, "*" ) )