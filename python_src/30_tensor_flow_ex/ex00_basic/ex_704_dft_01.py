# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

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
from numpy.fft import fft

# hide toolbar
mpl.rcParams['toolbar'] = 'None'

fig, ax= plt.subplots(1)

if 1 :
    x = np.linspace( 0, 3, 300 )
    # 5 + 2*cos(2*pi*x - pi/2) + 3cos(4*pi*x)
    y = 5 + 2*np.cos(0.5*pi*x - pi/2) + 3*np.cos(pi*x)
    ax.plot( x , y, label="sine(x)" , linestyle='dotted')
pass


x = np.array( [ 0, 1, 2, 3 ] )
f = np.array( [ 8, 4, 8, 0 ] )

ox = 0.03 #offset x
ax.bar( x + ox , f, width=0.02, color="green" )
ax.scatter( x + ox, f, label="Sampling", color="green" )

# 푸리에 변환
F = fft( f )
F_abs = np.absolute( F )
F_phase_deg = np.angle( F, deg=1 )
F_phase_rad = np.angle( F, deg=0 )

print( f"f = {f} " )
print( f"F = {F} " )
print( f"abs(F)  = {F_abs} " )
print( f"phase(F) deg = {F_phase_deg} " )

ox = -0.03 #offset x
ax.bar( x + ox, F_abs, width=0.02, color="blue" )
ax.scatter( x + ox, F_abs, label="FFT", color="blue" )

ax.xaxis.set_major_locator( MultipleLocator(base=1))
ax.yaxis.set_major_locator( MultipleLocator(base=2))

plt.legend()
plt.title( "Exmple sigonal for DFT" )
plt.xlabel("Time (seconds)")
plt.ylabel("Strength")

plt.show()