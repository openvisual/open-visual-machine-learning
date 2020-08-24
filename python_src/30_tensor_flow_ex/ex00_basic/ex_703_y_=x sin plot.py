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

# x 최소 값
x0 = -1*pi
# x 최대 값
x1 = 1*pi

# x 좌표값 들
x = np.linspace(x0, x1, 300 )
y = None

for n in range( 1, 10 ) :
    bn = -2*pow(-1, n)/n
    yn = bn*np.sin(n*x)/n

    label = f"{bn:.2g} sin({n}x)/{n}"
    if n > 5 :
        label = None
    elif bn == -1 :
        label = f"-sin({n}x)/{n}"
    pass

    ax.plot(x, yn, label=label, linestyle='dotted')

    if y is None :
        y = yn
    else :
        y += yn
    pass
pass

ax.plot( x , y, label=f"sum of sines", color="green" )
ax.plot( x , x, label=f"y = x", color="yellow", linestyle="dashed" )

ax.set_xlim( x0, x1 )

ax.xaxis.set_major_formatter(FuncFormatter( format_pi ))
ax.xaxis.set_major_locator( MultipleLocator(base=pi/2))
ax.yaxis.set_major_locator( MultipleLocator(base=1))

plt.title( "y = x Fourier Series" )
plt.xlabel("x")
plt.ylabel("sin(x)")

plt.grid( 1 )
plt.legend(loc="upper left")
plt.show()