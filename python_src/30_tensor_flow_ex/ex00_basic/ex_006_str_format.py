# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# 로그 예제
import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

''' 문자열 포맷 '''

# 문자열 합치기 (String concatenation)
a = "Good" + " morning " + str(1) + ", " + str(23.456)

# 문자열 포맷 지정자 %? (String format specifier)
b = "Good %s %d, %f" % ( "morning", 1, 23.456 )
b = "Good %s %03d, %5.1f" % ( "morning", 1, 23.456 )
c = "Good %s %-03d, %05.2f" % ( "morning", 1, 23.456 )

# format 함수
# %? 대신에 {:?} 을 사용한다.
d = "Good {:s} {:-03d}, {:05.2f}".format( "morning", 1, 23.456 )
e = "Good {:s} {:-03d}, {:05.2f}".format( "morning", -1, 23.456 )

# format 문자열
# - 문자열 앞에 f 지정자를 사용.
# - 문자열 안에서 변수/값을 사용.

m = "morning" ; one = 1; pct = 23.456

f = f"Good {'morning':s} {1:-03d}, {23.456:05.2f}"
g = f"Good {m:s} {one:-03d}, {pct:05.2f}"

print( "a = " + a )
print( "b = " + b )
print( "c = " + c )
print( "d = " + d )
print( "e = " + e )
print( "f = " + f )
print( "g = " + g )

''' -- 문자열 포맷 '''