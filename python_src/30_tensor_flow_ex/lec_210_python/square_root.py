import math

import logging
logging.basicConfig( format='%(levelname)-8s %(asctime)s %(filename)s %(lineno)d %(message)s', level=logging.DEBUG )

class SquareRoot :
    def find_square_root(self, n ) : 
        x = n
        y0 = 2
        y1 = 1
        while y1 != y0  :
            y0 = y1
            x = (x + y1)/2
            y1 = n/x
        pass
        
        return x
    pass
pass

print( "find a square root")
s = SquareRoot()
y = s.find_square_root( 4 ) 
print( "square root = %s" % y )