# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings 
warnings.filterwarnings('ignore',category=FutureWarning)

import logging as log
log.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO
    )

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 25,0.1)
fig, axis = plt.subplots(2)

plt.ylabel('sin(x)')
plt.xlabel('x')

axis[0].plot(np.sin(x))
axis[1].plot(np.cos(x))

plt.show()