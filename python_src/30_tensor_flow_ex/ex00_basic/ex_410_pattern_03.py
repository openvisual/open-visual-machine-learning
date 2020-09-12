# -*- coding: utf-8 -*-

for i in range( 1, 10 ):
    count = i if i < 6 else 10 - i

    '''
    count = i
    if i > 5 :
        count = 10 - i
    pass
    '''

    pattern = " * " * count

    print( pattern )
pass