# -*- coding:utf-8 -*-

to = 1_000 + 1
it_count = 0
idx = 1

length_list = []
area_list = []
a_list = []
b_list = []
c_list = []

for a in range(3, to) :
    for b in range(a + 1, to):
        for c in range( b + 1, to):
            it_count += 1
            if a < b < c and a*a + b*b == c*c :
                length = a + b + c
                area = a*b//2

                a_list.append(a)
                b_list.append(b)
                c_list.append(c)

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

x = range( len( area_list ) )

plt.plot(x, c_list, label="c")
plt.plot(x, b_list, label="b")
plt.plot(x, a_list, label="a")
plt.legend()
plt.show()

plt.bar( x, length_list, label="Length" )
plt.plot( x, area_list, label="Area" )

plt.legend()
plt.show()

# -- end