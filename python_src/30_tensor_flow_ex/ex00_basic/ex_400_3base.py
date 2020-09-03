base = 3

for i in range( 1, 20 ) :
    mok = i

    remainderList = []

    while mok > 0 :
        remainder = mok % base
        mok = mok // base
        # reverse list
        remainderList.append(remainder)
    pass

    #yungul = yungul[:: -1]
    oneNumber = 0
    zarisu = 1
    for i in remainderList :
        oneNumber = oneNumber + zarisu * i
        zarisu = zarisu * 10
    pass

    print( "결과", i, " = ", oneNumber, "(3) = ", remainderList[::-1])
pass