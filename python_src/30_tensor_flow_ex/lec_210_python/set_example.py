# coding=utf-8

# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# 합집합
# Output: {1, 2, 3, 4, 5, 6, 7, 8}
print(A | B)    

# 교집합
# Output: {4, 5}
print(A & B)

# 차집합
# Output: {1, 2, 3}
print(A - B)