# -*- coding:utf-8 -*-

to = 10_000
it_count = 0
i = 0

length_list = []
area_list = []
a_list = []
b_list = []
c_list = []

for a in range(3, to) :
    for b in range(a + 1, to):
        for c in range( b + 1, to):
            it_count += 1

            if c*c > a*a + b*b :
                # c*c 값이 a*a + b*b 보다 크게 되면 더 이상 의미 있는 c 의 값은 없다.
                # c 루프를 빠져 나감.
                break
            elif a*a + b*b == c*c :
                # 피타고라스 수를 찾음.
                length = a + b + c
                area = a*b//2

                a_list.append(a)
                b_list.append(b)
                c_list.append(c)

                length_list.append( length )
                area_list.append( area )

                print( f"[{100*a//to}% {i +1:d}] {a}, {b}, {c}, length = {length:,d}, area = {area:,d}")
                i += 1

                # a, b, c 쌍을 찾게 되면 c 루프를 빠져 나간다. 더 이상의 의미 있는 c는 없으므로
                break
            pass
        pass
    pass
pass

print( f"실행 횟수 : {it_count:,d}")

# 챠트 그리기
import matplotlib.pyplot as plt

# x 값을 생성함.
x = range( len( area_list ) )

# a, b, c 챠트 출력
plt.plot(x, c_list, label="c")
plt.plot(x, b_list, label="b")
plt.plot(x, a_list, label="a")
plt.legend()
plt.show()

# 길이, 면적 챠트 출력
plt.bar( x, length_list, label="Length" )
plt.plot( x, area_list, label="Area" )

plt.legend()
plt.show()

# -- end