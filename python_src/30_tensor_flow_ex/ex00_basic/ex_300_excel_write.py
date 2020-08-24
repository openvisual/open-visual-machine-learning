# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import xlsxwriter

# 엑셀 파일을 만드는 예제입니다.
# https://xlsxwriter.readthedocs.io/chart.html

print( "\nHello.... Good morning!\n" )

# 엑셀 파일(workbook)을 만들고, 엑셀 시트를 하나 추가함.
path = 'C:/temp/my_doc.xlsx'
workbook = xlsxwriter.Workbook( path )
worksheet = workbook.add_worksheet()

# 엑셀 시트에 추가할 데이터
expenses = (
    ['Rent', 1000],
    ['Gas',   100],
    ['Food',  300],
    ['Gym',    50],
)

# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

# Iterate over the data and write it out row by row.
for item, cost in expenses :
    worksheet.write(row, col,     item)
    worksheet.write(row, col + 1, cost)
    row += 1
pass

# Write a total using a formula.
worksheet.write(row, 0, 'Total')
worksheet.write(row, 1, '=SUM(B1:B4)')

workbook.close()

print( "Excel file(%s) was saved." % path )

print( "\nGood bye!" )

# -- 엑셀 파일을 만드는 예제입니다.