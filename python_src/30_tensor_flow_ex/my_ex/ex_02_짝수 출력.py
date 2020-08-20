# -*- coding: utf-8 -*-

# 1 ~ 100 사이의 짝수를 모두 출력하세요.

# for 문을 사용하는 것을 권장합니다.
print( "짝수 출력하기")

for i in range( 1, 101 ) :
    if i%2 == 0 :
        print( i )
    pass
pass

print( "Good bye!")