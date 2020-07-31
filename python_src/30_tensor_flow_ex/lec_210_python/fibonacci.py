class Fibonacci :
    def find_number(self, n): 
        if n == 1:
            return 1
        elif n == 0:   
            return 0            
        else:                      
            f_1 = self.find_number(n-1)
            f_2 = self.find_number(n-2)  
            return f_1 + f_2
        pass
    pass
pass 

print( "find fibonacci numbers")
f = Fibonacci() 
y = f.find_number( 3 )
print( "y = %s" % y )