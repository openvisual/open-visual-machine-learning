import math

class Pythagoras :

    def find_numbers_analytic( self ):
        print( "find_numbers_analytic" )
        for x in range( 1, 101 ): 
            for y in range( x + 1, 101 ):  
                real_z = math.sqrt( x*x + y*y ) 
                z = int( real_z )
                if z == real_z and z < 101 : 
                    print( "x = %s, y= %s, z= %s" %(x, y, z) )
                pass
            pass
        pass
    pass

    def find_numbers_numerical( self ):
        print( "find_numbers_numerical" )
        n = 1000
        for x in range( 1, n ) : 
            for y in range( x + 1, n ) :  
                for z in range( y , n ) :
                    if x*x + y*y == z*z : 
                        print( "x = %s, y= %s, z= %s" %(x, y, z) )
                    pass
                pass
            pass
        pass
    pass
pass

p = Pythagoras()
p.find_numbers_numerical() 

#--