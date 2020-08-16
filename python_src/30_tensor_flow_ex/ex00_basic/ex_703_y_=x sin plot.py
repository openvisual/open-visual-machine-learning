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

# x pi value format
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

x0 = -pi
x1 = pi

# 첫 번째 사인 함수
x = np.linspace(x0, x1, 300 )
sy_01 = np.sin(x)

ax.plot( x , sy_01, label="a = sine(x)" )

# 두 번째 사인 함수
sy_02 = 0.2*np.sin( 10*x )
ax.plot( x , sy_02, label="b = 0.2sine(10x)" )

# 첫 번 째 + 두 번 째 = 세번 째 사인 함수
sy_03 = sy_01 + sy_02

ax.plot( x , sy_03, label="c = a + b" , color="red")

ax.set_xlim( x0, x1 )

ax.xaxis.set_major_formatter(FuncFormatter( format_pi ))
ax.xaxis.set_major_locator( MultipleLocator(base=pi/2))
ax.yaxis.set_major_locator( MultipleLocator(base=1))

plt.title( "y = x Fourier Series" )
plt.xlabel("x")
plt.ylabel("sin(x)")

plt.grid( 1 )
plt.legend()
plt.show()