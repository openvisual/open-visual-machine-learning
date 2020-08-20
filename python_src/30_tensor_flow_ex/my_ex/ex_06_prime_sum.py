# -*- coding: utf-8 -*-
print( "Hello" )
prime_numbers = []

for number in range(2, 100) :
    isPrime = True

    for p in prime_numbers :
        if number%p == 0 :
            isPrime = False
            break
        pass
    pass

    if isPrime :
        prime_numbers.append(number)
    pass
pass

print( "소수들 = ", prime_numbers)
print( "소수의 합 = ", sum(prime_numbers))