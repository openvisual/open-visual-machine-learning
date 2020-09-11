# -*- coding: utf-8 -*-

col = 100

excel_column = ""

AZ_len = ord('Z') - ord('A') + 1

while col > 0:
    col, remainder = divmod(col - 1, AZ_len)

    c = chr(ord('A') + int(remainder))

    excel_column = c + excel_column
pass

if not excel_column:
    excel_column = "A"
pass

print( excel_column )

