# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings 
warnings.filterwarnings('ignore',category=FutureWarning)

import logging as log
log.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO
    )

from math import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

import numpy as np

# hide toolbar
mpl.rcParams['toolbar'] = 'None'

fig, ax= plt.subplots(1)

x = [ 0, 4*pi ]

sx = np.linspace(x[0], x[1], 100 )
sy = np.sin(sx)

print( sy )

ax.plot( sx , sy, label="sine" )

ax.set_xlim( x[0], x[1] )
ax.set_yticks( [ -1, 0, 1] )

def format( val, pos ) :
    vopi = val/pi

    if vopi == 0 :
        v = '0'
    elif int( vopi ) == vopi:
        v = f'{vopi:.0f} $\pi$'
    else :
        v = f'{vopi:.1f} $\pi$'
    pass

    return v
pass

ax.xaxis.set_major_formatter(FuncFormatter( format ))
ax.xaxis.set_major_locator( MultipleLocator(base=pi/2))

plt.title( "Sine signal" )
plt.xlabel("Time (seconds)")
plt.ylabel("Strength")

plt.grid( 1 )
plt.legend()
plt.show()