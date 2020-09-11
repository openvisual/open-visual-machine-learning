# -*- coding:utf-8 -*-

# 기계적 입실론 구하기

e = 1

while 1 + e/2 > 1 :
    e = e/2
pass

print( f"epsilon = {e}" )

import sys

e = sys.float_info.epsilon
print( f"epsilon float_info = {e}" )