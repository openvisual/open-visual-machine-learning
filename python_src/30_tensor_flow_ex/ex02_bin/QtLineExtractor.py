# -*- coding: utf-8 -*-

import logging as log
log.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO
    )

import os, sys
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QSettings, QPoint, QSize

from rsc.my_qt import *

from QtImageViewer import *


class MyQtApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__() # Call the inherited classes __init__ method

        uic.loadUi('./QtLineExtractor.ui', self)

        self.settings = QSettings('TheOneTech', 'line_extractor')

        self.imageViewers = []

        self.init_tab( self.tabWidgetLeft )
        self.init_tab( self.tabWidgetRight )

        self.tabWidgetLeft.currentChanged.connect( self.when_tab_widget_current_changed )

        self.exitBtn.clicked.connect( self.when_exitBtn_clicked )
        self.actionExit.triggered.connect(self.close_app)
        self.actionOpen.triggered.connect( self.when_openBtn_clicked )
        self.openBtn.clicked.connect( self.when_openBtn_clicked )

        self.startEvent()
    pass

    def when_openBtn_clicked(self, e):
        imageViewers = self.imageViewers

        if imageViewers :
            if imageViewers[0] is not None :
                imageViewer = imageViewers[0]
                fileName = imageViewer.loadImageFromFile()

                _, ext = os.path.splitext( fileName )
                ext = ext.lower()

                if fileName :
                    directory = os.path.dirname(fileName)
                    log.info( f"dir = {directory}" )

                    find_files = f"{directory}/*{ext}"
                    log.info( f"find_files={find_files}" )

                    import glob
                    files = glob.glob( find_files )

                    sel_file = None

                    fileNameBase = os.path.basename( fileName )

                    for file in files :
                        fileBase = os.path.basename( file )
                        log.info( f"fileBase = { fileBase }")

                        if fileBase > fileNameBase :
                            sel_file = file
                            break
                        pass
                    pass

                    log.info( f"sel_file = {sel_file}")

                    if sel_file is not None :
                        imageViewers[1].loadImageFromFile( sel_file )
                    pass
                pass
            pass
        pass
    pass

    def startEvent(self):
        settings = self.settings

        # 마지막 윈도우 크기 및 위치 로딩
        self.resize(self.settings.value("size", QSize(840, 640)))
        self.move(self.settings.value("pos", QPoint(50, 50)))
    pass

    def showEvent(self, e ):
        log.info( f"showEvent size={self.size()}" )

        return QtWidgets.QMainWindow.showEvent(self, e)
    pass

    def init_tab(self, tabWidget):
        settings = self.settings

        tabWidget.setCurrentIndex(0)

        tab = tabWidget.widget( 0 )

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setContentsMargins( 2, 2, 2, 2 )

        row = 0
        col = 0

        imageViewer = QtImageViewer(settings = settings)
        self.imageViewers.append(imageViewer)

        if 1 :
            fileNameLineEdit = QtWidgets.QLineEdit()
            fileNameLineEdit.setReadOnly( 1 )

            imageViewer.fileNameLineEdit = fileNameLineEdit

            horizontal = QtWidgets.QHBoxLayout()
            horizontal.addWidget( QtWidgets.QLabel( " File: ") )
            horizontal.addWidget( fileNameLineEdit )
            0 and horizontal.addItem( QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum) )

            gridLayout.addLayout( horizontal, row, col)
        pass

        row += 1
        gridLayout.addWidget( imageViewer, row, col )

        if 1 :
            horizontal = QtWidgets.QHBoxLayout()
            zoomIn = QtWidgets.QPushButton("Zoom In")
            zoomOut = QtWidgets.QPushButton("Zoom Out")
            fullExt = QtWidgets.QPushButton("Full Extent")
            original = QtWidgets.QPushButton("100%")
            horizontal.addWidget( zoomIn )
            horizontal.addWidget( zoomOut )
            horizontal.addWidget( fullExt )
            horizontal.addWidget( original )
            spacer = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            horizontal.addItem( spacer )

            row += 1
            gridLayout.addLayout( horizontal, row, col )
        pass

        tab.setLayout( gridLayout )

    pass

    def when_exitBtn_clicked(self, e):
        log.info( "when_exitBtn_clicked" )

        self.close()
    pass

    def closeEvent(self, e):
        log.info( "closeEvent")
        # 마지막 윈도우 크기 및 위치 저장 
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())

        e.accept()
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

