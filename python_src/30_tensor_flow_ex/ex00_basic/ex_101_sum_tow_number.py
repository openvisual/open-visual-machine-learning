import logging
logging.basicConfig(
    format='%(levelname)-8s %(asctime)s %(filename)s %(lineno)d %(message)s',
    level=logging.DEBUG
    )

def sum_between_two_number( x, y ) :  
    import xlsxwriter
    workbook = xlsxwriter.Workbook('sum_data.xlsx')
    worksheet = workbook.add_worksheet()

    t = x
    x = min( t, y )
    y = max( t, y ) 
    sum = 0
    row = 0 
    
    worksheet.write(row, 0, "x" )
    worksheet.write(row, 1, "sum" )
    row += 1

    x_array = []
    y_array = []
    for x in range( x, y + 1 ):
        sum = sum + x 
        x_array.append( x )
        y_array.append( sum )

        worksheet.write(row, 0, x )
        worksheet.write(row, 1, sum )

        row += 1
        logging.debug( " current x = %s, sum = %s" % ( x, sum ) )
    pass

    worksheet.write(row, 0, "sum" )
    worksheet.write(row, 1, sum ) 

    # Create a new chart object.
    chart = workbook.add_chart({'type': 'line'})
    # Add a series to the chart.
    chart.add_series({
        'categories': '=Sheet1!$A$2:$A$%s' % (row-1),
        'values': '=Sheet1!$B$2:$B$%s' % (row-1) 
    })
    chart.add_series({'values': '=Sheet1!$A$2:$B$%s' % (row-1) }
    )
    # Insert the chart into the worksheet.
    worksheet.insert_chart('D1', chart)

    workbook.close()

    show_chart = False 
    if show_chart : 
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np

        plt.plot(x_array, y_array)
        plt.show()
    pass

    return sum
pass

x = 1
y = 100

use_input = False 
if use_input : 
    x = int(input("Enter a number: "))
    y = int(input("Enter a number: "))
pass

sum = sum_between_two_number( x, y )

print ( "sum = %s" % sum )