# -*- coding: utf-8 -*-

import logging as log
log.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO
    )

import os, sys
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtWidgets import QApplication, QWidget

from rsc.my_qt import *

from QtImageViewer import *


class MyQtApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__() # Call the inherited classes __init__ method

        uic.loadUi('./QtBinApp.ui', self)

        # 탭 인덱스 설정

        self.init_tab( self.tabWidgetLeft )
        self.init_tab( self.tabWidgetRight )

        self.tabWidgetLeft.currentChanged.connect( self.when_tab_widget_current_changed )

        self.exitBtn.clicked.connect( self.when_exitBtnClicked )
        self.actionExit.triggered.connect(self.close_app)
    pass

    def init_tab(self, tabWidget):
        tabWidget.setCurrentIndex(0)

        tab = tabWidget.widget( 0 )

        lay = QtWidgets.QVBoxLayout( tab )
        m = 2
        lay.setContentsMargins( m, m, m, m)

        imageViewer = QtImageViewer()
        lay.addWidget( imageViewer )
    pass

    def when_exitBtnClicked(self, e):
        log.info( "when_exitBtnClicked" )

        self.close()
    pass

    def close_app( self ):
        log.info( "close app" )
        self.hide()
        sys.exit()
    pass # -- close_app

    def when_tab_widget_current_changed(self, index):
        log.info("when_tabWidget_currentChanged")
        pass
    pass # -- when_tab_widget_current_changed

    def show_image_on_image_viewer(self , image_viewer, image_path ):
        size = image_viewer.geometry()
        w = size.width()
        h = size.height()

        image_profile = QtGui.QImage(image_path)  # QImage object
        image_profile = image_profile.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio, transformMode=QtCore.Qt.SmoothTransformation)
        # To scale image for example and keep its Aspect Ration
        image_viewer.setPixmap(QtGui.QPixmap.fromImage(image_profile))
    pass
pass

if __name__ == '__main__':
    log.info( f"Pwd 1: {os.getcwd()}" )

    # change working dir to current file
    dir_name = os.path.dirname(__file__)
    if dir_name :
        log.info( f"dir name: {dir_name}" )
        os.chdir(dir_name)
        log.info( f"Pwd 2: {os.getcwd()}" )
    pass

    # Create an instance of QtWidgets.QApplication
    app = QtWidgets.QApplication(sys.argv)
    window = MyQtApp() # Create an instance of our class
    window.show()
    app.exec_() # Start the application
pass

