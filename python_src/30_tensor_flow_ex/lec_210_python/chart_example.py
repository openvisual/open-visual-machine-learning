import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x_array = np.linspace(0, 20, 10)
y_array = np.sin(x_array)

print( "x ========")
print( x_array )
print( "y ========")
print( y_array )

plt.plot(x_array, y_array)
plt.show()