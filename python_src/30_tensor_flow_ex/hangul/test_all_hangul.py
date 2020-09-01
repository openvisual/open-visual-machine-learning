# -*- coding: utf-8 -*-

# 한글 자모 분리병합 소스
# https://github.com/kaniblu/hangul-utils
# git clone https://github.com/kaniblu/hangul-utils han_utils

from han_utils.hangul_utils import split_syllable_char, split_syllables, join_jamos

# https://namu.wiki/w/%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C

# 유니코드에서 한글은 한자 다음으로 많은 코드를 차지하고 있는 문자다.
# 이것은 동아시아권에서 사용하는 문자로서는 두 번째로 많은 영역을 차지하는 것이다.
# 왜 저렇게 많냐면 현대 한국어 음절 조합과 한글 자모를 모두 집어넣었기 때문이다.

# http://xn--bj0bv9kgwxoqf.org/%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C
'''
    블록 범위	내용	기호	기호수
    U+1100 ~ U+11FF	한글 자모	    ㄱ ~ ᇿ	256
?   U+3000 ~ U+303F	한중일 문장부호	・ ~ ー	2
    U+3130 ~ U+318F	한글 호환 자모	ㄱ ~ ㆎ	94
?   U+A960 ~ U+A97F	한글 확장 자모A	ꥠ ~ ꥼ	29
?   U+AC00 ~ U+D7AF	한글 글자 마디	가 ~ 힣	11,172
?   U+D7B0 ~ U+D7FF	한글 확장 자모B	ힰ ~ ퟻ	72
?   U+FF00 ~ U+FFEF	반각 자모	    ﾡ ~ ￜ	52
'''

# '가' 한글의 유니코드값을 정수로 변환
kor_unicode_start = ord("가")
# '힣' 한글의 유니코드 값을 정수로 변환
kor_unicode_end = ord("힣")

# 한글 분할/합병 테스트 시작 글자
i = kor_unicode_start

# 테스트 횟수
idx = 0

while i <= kor_unicode_end :
    # 정수 유니코드 값으로 부터 한글 유니코드로 변환
    kor = chr( i )
    # 한글 자소 분리
    split = split_syllables( kor )
    # 한글 자수 병합
    merge = join_jamos( split )

    # 병합된 한글과 원래 한글과 같은 지 판변함.
    is_same = ( kor == merge )

    print( f"[{idx:05d}]: 유니코드 = U+{i:X}, 정수 = {i}, 한글 = {kor}, 분할 = {split}, 합병 = {merge}, 일치 = {is_same}" )

    # 분할 합볃 되지 않았을 때, 에러 메시지 출력 .
    if not is_same :
        print( "분할 합병이 되지 않는 글자가 발견되었습니다.")
        break
    pass

    # 유니코드 다음 글자 설정
    i += 1
    # 테스트 회수 증가
    idx += 1
pass
