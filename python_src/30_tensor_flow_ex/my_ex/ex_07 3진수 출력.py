# -*- coding: utf-8 -*-

# 1 ~ 100 사이의 모든 수를 3진수로 출력하세요.

print( "Hello" )

# 이 해수 차장님의 문제 풀이 입니다.

# 아래 줄 NOTATION을 삭제하고 풀어 보세요.
# 문자열 또는 문자형을 사용하지 않고 풀어 보세요.

NOTATION = '0123456789ABCDEF'

def numeral_system(number, base):
    q, r = divmod(number, base)
    n = NOTATION[r]
    return numeral_system(q, base) + n if q else n
pass

for x in range( 0, 11 ) :
    result = numeral_system(x, 3)
    print( result )
pass


