# -*- coding:utf-8 -*-

for i in range( 9 + 1 ):
    count = ( i + 1 ) if i < 5 else 9 - i

    '''
    count = i + 1
    if i >= 5:
        count = 9 - i
    pass
    '''

    pattern = ( " * "*count ).center( 13 )

    print( pattern )
pass