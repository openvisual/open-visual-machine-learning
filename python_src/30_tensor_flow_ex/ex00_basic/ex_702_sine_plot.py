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
import matplotlib.colors as mcolors
import numpy as np

f, ax= plt.subplots(1)

x = np.linspace(0, 2*pi, 100 )
y = np.sin(x)

ax.plot( x , y )

ax.set_xlim(0, 2*pi)

ax.set_xticks( [ 0, pi, 2*pi ] )
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
#ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))

plt.grid( 1 )
plt.show()