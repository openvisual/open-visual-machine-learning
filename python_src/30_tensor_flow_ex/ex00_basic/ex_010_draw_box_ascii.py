# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

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
print( "박스" )
print( box )

lines = box.split( "\n" )

print()
print( "Lines" )
for i, line in enumerate( lines ) :
    print( i, " ", line )
pass
