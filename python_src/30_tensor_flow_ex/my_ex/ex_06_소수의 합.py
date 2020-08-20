# -*- coding: utf-8 -*-

# 1 ~ 100 사이의 모든 소수들의 합을 구하세요.

print( "Hello" )

prime_numbers = []

for number in range(2, 100) :
    is_prime = True

    for p in prime_numbers :
        if number%p == 0 :
            is_prime = False
            break
        pass
    pass

    if is_prime :
        prime_numbers.append(number)
    pass
pass

print( "소수들 = ", prime_numbers)
print( "소수의 합 = ", sum(prime_numbers))