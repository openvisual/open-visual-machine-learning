# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings 
warnings.filterwarnings('ignore', category=FutureWarning)

# if else 문장을 간단히 구사하는 기교입니다.

w = 10
h = 20

a = 0

if w < h :
    a = h
else :
    a = w
pass

b = [ w, h ][ w < h ]

c = ( w, h )[ w < h ]

d = h if w < h else w

print( "a =", a )
print( "b =", b )
print( "c =", c )
print( "d =", d )

# -- if else 예제