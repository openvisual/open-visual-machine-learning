# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import logging as log
log.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO
    )

import xlsxwriter

# 엑셀 파일을 만드는 예제입니다.
# https://xlsxwriter.readthedocs.io/chart.html

print( "\nHello.... Good morning!\n" )

# 엑셀 파일(workbook)을 만들고, 엑셀 시트를 하나 추가함.
file_path = 'C:/temp/my_chart.xlsx'
# 엑셀 파일 만들기
workbook = xlsxwriter.Workbook(file_path)
# 엑셀 시트 추가
worksheet = workbook.add_worksheet()

# 엑셀 시트에 추가할 데이터
expenses = (
    ['Rent', 1000, 0.2],
    ['Gas',   100, 0.3],
    ['Food',  300, 0.4],
    ['Gym',    50, 0.5],
)

# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

# 엑셀 시트에 열과 행에 데이터를 추가한다.
for item, cost, ratio in expenses :
    worksheet.write(row, 0,     item)
    worksheet.write(row, 1, cost)
    worksheet.write(row, 2, ratio)
    row += 1
pass

# 함계를 엑셀 함수를 이용하여 구한다.
worksheet.write(row, 0, 'Total')
worksheet.write(row, 1, '=SUM(B1:B4)')

if 1 : # 챠트 추가
    chart = workbook.add_chart({'type': 'line'})

    # Add a series to the chart.
    chart.add_series({
        'categories': '=Sheet1!A1:A4',
        'values': '=Sheet1!B1:B4'
    })

    # Add a series to the chart.
    chart.add_series({'values': '=Sheet1!C1:c4'} )

    # Insert the chart into the worksheet.
    worksheet.insert_chart('B7', chart)
pass

# 파일 이미지 추가.
worksheet.insert_image('B25', 'python.png' )

# 웹 이미지 파일 추가

from io import BytesIO
from urllib.request import urlopen

url = "https://expertsystem.com/wp-content/uploads/2017/03/machine-learning-definition.jpeg"
image_data = BytesIO(urlopen(url).read())
worksheet.insert_image('G25', url, {'image_data': image_data})

workbook.close()

print( "Excel file(%s) was saved." % file_path)

''' 엑셀 파일 열기 '''
import webbrowser as wb
wb.open(file_path)

print( "\nGood bye!" )

# -- 엑셀 파일을 만드는 예제입니다.