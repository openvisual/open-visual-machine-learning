# -*- coding: utf-8 -*-

# 1 ~ 100 사이의 3의 배수를 모두 출력하세요.

# for 문을 사용하는 것을 권장합니다.
print( "3의 배수")

for i in range( 1, 101 ) :
    if i%3 == 0 :
        print( i )
    pass
pass

print()
print( "다른 방법" )
for i in range( 3, 101, 3 ) :
    print( i )
pass

print( "Good bye!")