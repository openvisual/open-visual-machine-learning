# coding=utf-8

import xlsxwriter

# 엑셀 파일을 만드는 예제입니다.

print( "\nHello.... Good morning!\n" )

# 엑셀 파일(workbook)을 만들고, 엑셀 시트를 하나 추가함.
path = 'C:/temp/my_chart.xlsx'
workbook = xlsxwriter.Workbook( path )
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

workbook.close()

print( "Excel file(%s) was saved." % path )

''' 폴더 열기 '''

import webbrowser as wb
wb.open( path )

print( "\nGood bye!" )

# -- 엑셀 파일을 만드는 예제입니다.