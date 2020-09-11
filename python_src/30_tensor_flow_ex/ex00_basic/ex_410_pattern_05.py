# -*- coding: utf-8 -*-

for i in range( 9 + 1 ):
    count = ( i + 1 ) if i < 5 else 9 - i

    pattern = ( " *" * count ).center( 10 )*5

    print( pattern )
pass