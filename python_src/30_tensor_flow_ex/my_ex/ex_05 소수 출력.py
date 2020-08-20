# -*- coding: utf-8 -*-

# 1 ~ 100 사이의 소수를 모두 출력하세요.

print( "소수" )

prime_numbers = []
for i in range( 2, 101 ) :
    is_prime = True
    for p in prime_numbers :
        if i%p == 0 :
            is_prime = False
            break
        pass
    pass

    if is_prime :
        prime_numbers.append( i )
pass

print( prime_numbers )

print( "Good bye!")