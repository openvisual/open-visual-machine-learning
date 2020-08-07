# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

# 아스키 코드로 박스 그리기

m = "═ ║ ╔ ╗ ╠ ╣ ╝ ╚"

print()
print( "m" )
print( m )

box = '''
╔════════════════════════════╗
║                            ║
╠════════════════════════════╣
║                            ║
╚════════════════════════════╝
'''
box = box.strip()

print()
print( "A Box")
print( box )

lines = box.split( "\n" )

print()
print( "Lines" )
for i, line in enumerate( lines ) :
    print( i, " ", line )
pass
