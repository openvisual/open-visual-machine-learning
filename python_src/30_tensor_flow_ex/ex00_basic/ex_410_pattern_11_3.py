# -*- coding: utf-8 -*-

for i in range( 1, 10 ):
    data = ""
    for k in range( 0, i ) :
        data += f"{k + 1}"
    pass

    data = data[::-1] + data[1:]

    print( data.center( 17 ) )
pass