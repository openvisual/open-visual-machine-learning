# -*- coding:utf-8 -*-

for i in range( 1, 10 + 1 ):
    if i <= 5 :
        pattern = " * " * i
    else :
        pattern = " * " * ( 10 - i )
    pass

    print( pattern )
pass