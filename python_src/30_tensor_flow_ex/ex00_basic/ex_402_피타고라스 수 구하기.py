# -*- coding:utf-8 -*-

to = 101
it_count = 0
idx = 1
for a in range(3, to ) :
    for b in range(a + 1, to):
        for c in range( b + 1, to):
            it_count += 1
            if a < b < c and a*a + b*b == c*c :
                length = a + b + c
                area = int( a*b/2 )
                print( f"[{idx:d}] {a}, {b}, {c} , length={length:,d}, area={area:,d}" )
                idx += 1
            pass
        pass
    pass
pass

print( f"실행 횟수 : {it_count:,d}")