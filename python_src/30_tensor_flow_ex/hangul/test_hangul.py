# -*- coding: utf-8 -*-

# 한글 자모 분리병합 소스
# https://github.com/kaniblu/hangul-utils
# git clone https://github.com/kaniblu/hangul-utils hanutils

from hanutils.hangul_utils import split_syllable_char, split_syllables, join_jamos

print(split_syllable_char(u"안"))

print(split_syllables(u"안녕하세요"))