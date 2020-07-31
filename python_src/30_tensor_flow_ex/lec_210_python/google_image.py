# coding=utf-8
import logging

logging.basicConfig( format='%(levelname)-8s %(asctime)s %(filename)s %(lineno)d %(message)s', level=logging.DEBUG )

class GoogleImage :
    def __init__(self) :
        pass
    pass

    def saveImage(self, key_word ) :
        import urllib.parse
        key_url = urllib.parse.urlencode( { "q" : key_word } )
        # import libraries
        from bs4 import BeautifulSoup 
        # specify the url
        html_url = "https://www.google.com/search?tbm=isch&%s" % key_url 

        # 이미지 저장 폴더 생성
        import os
        desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        image_path = desktop_path + "\\googoe_image" + "\\" + key_word  

        # 폴더가 없으면 생성함.
        if not os.path.exists( image_path ):
            os.makedirs( image_path )
        pass

        # query the website and return the html to the variable ‘page’
        logging.info( "Getting a html data from %s" % html_url )

        from urllib.request import Request, urlopen
        hdr = {'User-Agent': 'Mozilla/5.0'}
        req = Request( html_url,headers=hdr)
        html_page = urlopen(req)  
        html_src = html_page.read()

        logging.info( "Done. Getting a html data from %s" % html_url )

        # parse the html using beautiful soup and store in variable `soup`
        html = BeautifulSoup( html_src, "html.parser" , from_encoding="euc-kr" )

        id_search_div = html.find( "div", { "id": "search" } )
        img_list = id_search_div.find_all( "img" )

        idx = 0 
        for img in img_list :
            img_url = img[ "src" ]
            
            logging.info( "[%02d] img src = %s" % ( idx, img[ "src" ] ) )

            # 이미지 가져와서 저장하기
            import shutil
            import requests

            response = requests.get( img_url, stream=True)
            file_name = "%s\\google_img_%s.png" % ( image_path, idx )
            with open( file_name, 'wb') as file:
                shutil.copyfileobj(response.raw, file) 
            pass

            idx += 1 
        pass

    pass

pass

if __name__ == '__main__' :   
    googleImage = GoogleImage()
    key_word_list = [ "독도", "대한민국", "고려인" ]
    for key_word in key_word_list : 
        googleImage.saveImage( key_word )
    pass
pass
