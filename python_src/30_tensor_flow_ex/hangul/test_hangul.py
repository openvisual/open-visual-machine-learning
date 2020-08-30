# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# 로그 예제
import logging as log
log.basicConfig(
    format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO
    )

from hanutils.hangul_utils import split_syllable_char, split_syllables, join_jamos

print(split_syllable_char(u"안"))

print(split_syllables(u"안녕하세요"))