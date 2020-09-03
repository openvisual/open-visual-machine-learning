base = 3
for number in range(1, 20) :
    mok = number
    remainderList = []
    while mok > 0 :
        remainder = mok % base
        mok = mok // base
        remainderList.append( remainder )
    pass

    oneNumber = 0
    zarisu = 1
    for i in remainderList :
        oneNumber += zarisu * i
        zarisu = zarisu * 10
    pass

    print( "결과", number, " = ", oneNumber, "(3) = ", remainderList[::-1])
pass