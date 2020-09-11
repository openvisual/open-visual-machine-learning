# -*- coding:utf-8 -*-

for i in range( 1, 10 ):
    if i < 6 :
        pattern = " * " * i
    else :
        pattern = " * " * ( 10 - i )
    pass

    print( pattern )
pass