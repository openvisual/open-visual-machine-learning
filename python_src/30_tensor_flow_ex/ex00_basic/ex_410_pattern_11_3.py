# -*- coding: utf-8 -*-

for i in range( 1, 1 + 9 + 1 ):
    data = ""
    for k in range( 1, i ) :
        data += f"{k}"
    pass

    data = data[::-1] + data[1:]

    print( data.center( 17 ) )
pass