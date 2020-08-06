# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

# 함수의 실행 시간 출력

# time을 사용한 예제 / 미추천 / 코드가 복잡함.
from time import time
start = time()
# your script here
end = time()
print(f'It took {end - start} seconds!')

# profile annotation 사용 추천 / 코드가 간단함.
# -- 사용법
# @profile
# def your_function(...):
#       ....
#
# your_function( ... )
# print_prof_data()

import time
from functools import wraps

PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        pass

        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling
pass # -- profile(fn)

def print_prof_data():
    for fname, data in PROF_DATA.items() :
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        fmt = "Function[ %s ] called %d times. Exe. time max: %.3f, average: %.3f"
        log.info( fmt % (fname, data[0], max_time, avg_time) )
    pass

    PROF_DATA.clear()
pass # -- print_prof_data()

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
pass # -- clear_prof_data

''' --- profile functions '''

# 사용법

@profile
def myFun():
    z = 0
    for x in range( 1, 100 ) :
        for y in range(1, 100):
            z += x*y

            log.info( "{:,}*{:,} = {:,}".format( x, y ,z) )
        pass
    pass

    return z
pass

myFun()

print_prof_data()

# --함수의 실행 시간 출력