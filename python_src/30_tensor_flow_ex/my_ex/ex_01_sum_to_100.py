# -*- coding: utf-8 -*-

# 1 ~ 100 까지의 모든 정수들의 합을 구하세요.

# for 문을 사용하는 것을 권장합니다.
print( "Hello")

my_sum = 0

for i in range( 1, 101 ) :
    my_sum = my_sum + i
pass

print( "합계 = " , my_sum )

# 다른 방법
print()
print( "Good morning" )

numbers = range( 1, 101 )
my_sum2 = sum( numbers )

print( "합계 2 = " , my_sum2 )