# coding=utf-8

import logging
logging.basicConfig( format='%(levelname)-8s %(asctime)s %(filename)s %(lineno)d %(message)s', level=logging.DEBUG )

class WebScrape :

    def __init__(self) :
        pass
    pass

    def doScrape(self) : 
        import xlsxwriter
        workbook = xlsxwriter.Workbook( "오늘의 증권시세.xlsx" )
        worksheet = workbook.add_worksheet()
        row = 0
        col = 0 

        line = "\n" + "#"*80 + "\n"

        # import libraries
        from bs4 import BeautifulSoup

        # specify the url
        html_url = "http://vip.mk.co.kr/newSt/rate/item_all.php"

        # query the website and return the html to the variable ‘page’

        logging.info( "Getting a html data from %s" % html_url )

        from urllib.request import urlopen
        html_page = urlopen( html_url )
        html_src = html_page.read()

        logging.info( "Done. Getting a html data from %s" % html_url )

        # parse the html using beautiful soup and store in variable `soup`
        soup = BeautifulSoup( html_src, "html.parser" , from_encoding="euc-kr" )

        print( "title = %s" % soup.title ) 
        print( "title.string = %s" % soup.title.string )
        table = soup.table        
        table_subs = table.find_all( "table" )
        print( line )         
        jisu = table_subs[ 1 ]
        trs = jisu.find_all( "tr" )
        for tr in trs :
            tds = tr.find_all( "td" )
            col = 0 
            for td in tds : 
                print( "td = %s" % td.string )
                worksheet.write(row, col, td.string )
                col += 1
            pass
            row += 1
        pass 

        jisu = table_subs[ 4 ]
        trs = jisu.find_all( "tr" )
        for tr in trs :
            tds = tr.find_all( "td" )
            col = 0 
            for td in tds : 
                print( "td = %s" % td.string )
                worksheet.write(row, col, td.string )
                col += 1
            pass
            row += 1
        pass 
        print( line )
        workbook.close() 
    pass

pass

if __name__ == '__main__' :   
    web_scrape = WebScrape()
    web_scrape.doScrape()
pass