# -*- coding:utf-8 -*-

for i in range( 1, 10 ):
    count = i if i < 6 else 10 - i

    pattern = ( " *"*count ).center( 9 )

    print( pattern )
pass