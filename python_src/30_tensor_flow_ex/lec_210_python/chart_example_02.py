# coding=utf-8
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def y( x ) :
    return x*x - 3*x -100
pass

x_array = np.linspace( -20, 20, 100 )
y_array = []
for x in x_array :
    yv = y( x )
    y_array.append( yv )
pass

plt.plot(x_array, y_array, 'r--', linewidth=1 )
 
# 추가 

plt.xlabel( 'x' )
plt.ylabel( 'y' )
plt.title( "y=x*x - 3x -100") 
plt.grid(True)
plt.show()