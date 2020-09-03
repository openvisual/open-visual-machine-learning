# -*- coding:utf-8 -*-

to = 1000 + 1
it_count = 0
idx = 1

length_list = []
area_list = []
for a in range(3, to ) :
    for b in range(a + 1, to):
        for c in range( b + 1, to):
            it_count += 1
            if a < b < c and a*a + b*b == c*c :
                length = a + b + c
                area = int( a*b/2 )

                length_list.append( length )
                area_list.append( area )
                print( f"[{idx:d}] {a}, {b}, {c}, length={length:,d}, area={area:,d}" )
                idx += 1
            pass
        pass
    pass
pass

print( f"실행 횟수 : {it_count:,d}")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

x = range( len( area_list ) )

plt.bar( x, length_list, color="green" , label="Length" )
plt.plot( x, area_list, color="blue" , label="Area" )

plt.legend()

plt.show()
pass