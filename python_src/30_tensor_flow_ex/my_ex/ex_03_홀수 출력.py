# -*- coding: utf-8 -*-

# 1 ~ 100 사이의 홀수를 모두 출력하세요.

print( "홀수 출력하기")

for i in range( 1, 101 ) :
    if i%2 == 1 :
        print( i )
    pass
pass

print()
print( "다른 방법" )

for i in range( 1, 101 , 2 ) :
    print( i )
pass

print( "Good bye!")