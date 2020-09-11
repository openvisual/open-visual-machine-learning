# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# 로그 예제
import logging as log
log.basicConfig(
    format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO
    )

# 로그를 남기는 예제입니다.
log.info( "Hello" )
log.info( "Good bye!" )

# -- 로그 예제