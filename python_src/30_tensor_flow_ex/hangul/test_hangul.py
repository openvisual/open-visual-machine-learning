# -*- coding: utf-8 -*-

# 한글 자모 분리병합 소스
# https://github.com/kaniblu/hangul-utils
# git clone https://github.com/kaniblu/hangul-utils han_utils

from han_utils.hangul_utils import split_syllable_char, split_syllables, join_jamos

print(split_syllable_char("안"))

print(split_syllables("안녕하세요"))

sentence = "앞 집 팥죽은 붉은 팥 풋팥죽이고, 뒷집 콩죽은 햇콩 단콩 콩죽.우리 집 깨죽은 검은 깨 깨죽인데 사람들은 햇콩 단콩 콩죽 깨죽 죽먹기를 싫어하더라."

s = split_syllables(sentence)

print( "s = ", s )

s2 = join_jamos(s)

print( "s2 = ", s2 )

print( sentence == s2 )