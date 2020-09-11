# -*- coding:utf-8 -*-

for i in range(1, 6):
    for j in range(i):
        print(i, end = '')
    pass
    print()
pass

print()
# 수정
for i in range( 5 ):
    pattern = f"{i+1}"*(i+1)
    #pattern *= (i + 1)
    #pattern = pattern*(i+1)
    print( pattern )
pass