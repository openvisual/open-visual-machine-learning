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

class MyQtApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__() # Call the inherited classes __init__ method

        uic.loadUi('./QtBinApp.ui', self)

        tabWidget = self.tabWidget

        if tabWidget.count() > 0 :
            # init first tab
            tab = tabWidget.widget( 0 )
            uic.loadUi('./TabContent.ui', tab)
        pass

        if tabWidget.count() > 0 :
            # set last tab text as +
            tabLast = QWidget()

            tabWidget.addTab(tabLast, "+")

            tabWidget.setCurrentIndex(0)
        pass

        self.appendingTab = False

        self.tabWidget.currentChanged.connect( self.when_tab_widget_current_changed )
        self.appendTab.clicked.connect( self.when_append_tab_clicked )
    pass

    def when_tab_widget_current_changed(self, index):
        log.info("when_tabWidget_currentChanged")

        tabWidget = self.tabWidget

        if self.appendingTab :
            log.info( "appendingTab" )
        elif index == tabWidget.count() - 1 :
            self.appendingTab = True
            self.when_append_tab_clicked()
            self.appendingTab = False
        pass
    pass # -- when_tab_widget_current_changed

    def when_append_tab_clicked(self):
        log.info( "when_append_tab_clicked" )

        tabWidget = self.tabWidget

        tabLast = tabWidget.widget( tabWidget.count() - 1 )

        tabWidget.removeTab( tabWidget.count() - 1 )

        tab = QWidget()

        uic.loadUi( './TabContent.ui', tab )

        tabWidget.addTab( tab, f"Tab {tabWidget.count() + 1}" )

        imageViewer = tab.imageViewer
        image_path = "./python.png"

        self.show_image_on_image_viewer( imageViewer, image_path )

        tabWidget.addTab( tabLast, "+" )
    pass # -- when_append_tab_clicked

    def show_image_on_image_viewer(self , image_viewer, image_path ):
        size = image_viewer.geometry()
        w = size.width()
        h = size.height()

        image_profile = QtGui.QImage(image_path)  # QImage object
        image_profile = image_profile.scaled(w, h,
                          aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                          transformMode=QtCore.Qt.SmoothTransformation)
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

