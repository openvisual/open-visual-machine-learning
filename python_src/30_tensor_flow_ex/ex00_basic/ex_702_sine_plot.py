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

# hide toolbar
mpl.rcParams['toolbar'] = 'None'

# pi 포맷 함수
def format_pi( val, pos ) :
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

fig, ax= plt.subplots(1)

x0 = 0
x1 = 4*pi

# 첫 번째 사인 함수
x = np.linspace(x0, x1, 300 )
y_01 = np.sin(x)

ax.plot( x , y_01, label="a = sine(x)", color = "blue" )

# 두 번째 사인 함수
y_02 = 0.2*np.sin( 10*x )
ax.plot( x , y_02, label="b = 0.2sine(10x)", color = "green" )

# 첫 번 째 + 두 번 째 = 세번 째 사인 함수
y_03 = y_01 + y_02

ax.plot( x , y_03, label="c = a + b" , color="red")

ax.set_xlim(x0, x1)
#ax.set_yticks( [ -1, 0, 1] )

ax.xaxis.set_major_formatter(FuncFormatter( format_pi ))
ax.xaxis.set_major_locator( MultipleLocator(base=pi/2))
ax.yaxis.set_major_locator( MultipleLocator(base=1))

plt.title( "Sine signal" )
plt.xlabel("Time (seconds)")
plt.ylabel("Strength")

plt.grid( 1 )
plt.legend()
plt.show()