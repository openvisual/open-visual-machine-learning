# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

# 메인 함수 예제 입니다.

class MyClass :
    def __init__(self):
        log.info( "MyClass" )
    pass
pass

def my_fun() :
    log.info( "Hellow...")
pass

if __name__ == '__main__':
     myClass = MyClass()

     my_fun()
pass # -- main

