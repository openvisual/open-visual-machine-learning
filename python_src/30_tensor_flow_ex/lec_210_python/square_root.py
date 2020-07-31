import math

import logging
logging.basicConfig( format='%(levelname)-8s %(asctime)s %(filename)s %(lineno)d %(message)s', level=logging.DEBUG )

class SquareRoot :
    def find_square_root(self, n ) : 
        x = n 
        y = 1
        e = 0.000001
        while x - y > e : 
            x = (x + y)/2
            y = n/x
        pass
        
        return x
    pass
pass

print( "find a square root")
s = SquareRoot()
y = s.find_square_root( 4 ) 
print( "square root = %s" % y )