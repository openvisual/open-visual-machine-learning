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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

import matplotlib.colors as mcolors
import numpy as np

f, ax= plt.subplots(1)

x = np.linspace(0, 2*pi, 100 )
y = np.sin(x)

ax.plot( x , y )

ax.set_xlim(0, 2*pi)

ax.set_xticks( [ 0, pi, 2*pi ] )
#x_labels = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$']
#ax.set_xticklabels(x_labels)
ax.set_yticks( [ -1, 0, 1] )
ax.xaxis.set_major_formatter(FuncFormatter( lambda val,pos: f'{val/pi:.1f} $\pi$' if val !=0 else '0'))
ax.xaxis.set_major_locator( MultipleLocator(base=pi/2))

plt.grid( 1 )
plt.show()