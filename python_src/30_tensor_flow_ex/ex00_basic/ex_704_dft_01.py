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
from numpy.fft import fft

# hide toolbar
mpl.rcParams['toolbar'] = 'None'

fig, ax= plt.subplots(1)

x = np.array( [ 0, 1, 2, 3 ] )
f = np.array( [ 8, 4, 8, 0 ] )

ax.plot( x , f, label="Discrete Signal" )

F = fft( x )
print( F )

ax.xaxis.set_major_locator( MultipleLocator(base=1))
ax.yaxis.set_major_locator( MultipleLocator(base=2))

plt.legend()
plt.title( "Exmple sigonal for DFT" )
plt.xlabel("Time (seconds)")
plt.ylabel("Strength")

plt.show()