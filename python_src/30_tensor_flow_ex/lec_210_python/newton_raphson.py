# coding=utf-8

# 입실론 : 아주 작은 값
e = 0.00001 

# 함수 정의
def f(x) :
   return x*x - 3*x -100
pass

# 기울기 구하는 함수
def derivative(x) :
   der = (f(x + e) - f(x)) / e 
   return der 
pass

def main() : 
    # 첫 교차점
    x = 6.0
    y = 1000000

    # 몇 번 해를 구했는 지를 나타냄.
    index = 0

    # y 값이 0 에 가까울 때 까지 해를 반복적으로 구함.
    while abs(y) >= e :
        y = f(x)  
        
        print( "[%03d] f(%.20f) = %.20f" % ( index, x, y) )
        
        # 다음 교차점
        x = x - (y / derivative(x))
        
        index += 1
    pass
pass

if __name__ == '__main__' :   
    main()
pass
