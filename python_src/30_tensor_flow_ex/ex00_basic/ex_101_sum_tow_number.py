# coding=utf-8

import logging
logging.basicConfig( format='%(levelname)-8s %(asctime)s %(filename)s %(lineno)d %(message)s', level=logging.DEBUG )

print( "Hello.... Good morning!")

path = "C:/temp/sum_data.xlsx"

def sum_between_two_number( x, y ) :  
    import xlsxwriter

    global path

    workbook = xlsxwriter.Workbook( path )
    worksheet = workbook.add_worksheet()

    sum = 0
    row = 0 
    
    worksheet.write(row, 0, "x" )
    worksheet.write(row, 1, "sum" )
    row += 1

    for i in range( x, y + 1 ):
        sum = sum + i

        worksheet.write(row, 0, i )
        worksheet.write(row, 1, sum )

        row += 1
        logging.debug( " current x = %s, sum = %s" % ( i, sum ) )
    pass

    worksheet.write(row, 0, "sum" )
    worksheet.write(row, 1, sum ) 

    # Create a new chart object.
    chart = workbook.add_chart({'type': 'line'})
    # Add a series to the chart.
    chart.add_series({
        'categories': '=Sheet1!A2:A%s' % (row-1),
        'values': '=Sheet1!B2:B%s' % (row-1)
    })
    # Insert the chart into the worksheet.
    worksheet.insert_chart('D2', chart)

    workbook.close()

    return sum
pass

x = 1
y = 100

use_input = 1
if use_input : 
    x = int(input("Enter a number: "))
    y = int(input("Enter a number: "))
pass

sum = sum_between_two_number( x, y )

print( "sum = %s" % sum )

import webbrowser as wb
wb.open( path )

print( "Good bye!")