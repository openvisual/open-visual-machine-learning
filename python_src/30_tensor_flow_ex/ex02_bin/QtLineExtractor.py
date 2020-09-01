# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, sys, inspect
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtWidgets import QApplication, QWidget, QAction
from PyQt5.QtCore import QSettings, QPoint, QSize, Qt

from rsc.my_qt import *
from QtImageViewer import *

class QtLineExtractor(QtWidgets.QMainWindow):
    def __init__(self):
        # Call the inherited classes __init__ method
        super(QtLineExtractor, self).__init__()

        uic.loadUi('./QtLineExtractor.ui', self)

        self.settings = QSettings('TheOneTech', 'line_extractor')

        self.imageViewers = []

        self.init_tab( self.tabWidgetLeft )
        self.init_tab( self.tabWidgetRight )

        self.progressBar.setValue( 0 )
        self.progressBar.setEnabled( 0 )

        time = QtCore.QTime.currentTime()
        text = time.toString('hh:mm:ss')
        self.durationLcdNumber.display( text )

        self.tabWidgetLeft.currentChanged.connect( self.when_tab_widget_current_changed )

        self.exitBtn.clicked.connect( self.when_exitBtn_clicked )
        self.actionExit.triggered.connect(self.close_app)
        self.actionOpen.triggered.connect( self.when_openBtn_clicked )
        self.openBtn.clicked.connect( self.when_openBtn_clicked )
        self.lineExtract.clicked.connect( self.when_lineExtract_clicked )

        self.buildOpenRecentFilesMenuBar()

        self.startEvent()
    pass

    def load_file(self, fileName):
        pass
    pass

    def buildOpenRecentFilesMenuBar(self):
        menuOpen_Recent = self.menuOpen_Recent

        settings = self.settings

        recent_file_list = settings.value( 'recent_file_list', [], str)

        menuOpen_Recent.setEnabled( len( recent_file_list ) > 0 )

        menuOpen_Recent.clear()

        for i, fileName in enumerate( recent_file_list ):
            log.info( f"fileName = {fileName}" )
            action = QAction( fileName , menuOpen_Recent )
            action.fileName = fileName
            menuOpen_Recent.addAction( action )

            action.triggered.connect( lambda val : self.when_recentFileAction_clicked(action.fileName) )
        pass
    pass # -- buildOpenRecentFilesMenuBar

    def when_recentFileAction_clicked(self, fileName ):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        self.when_openBtn_clicked( e=None, fileName=fileName )
    pass # -- when_recentFileAction_clicked

    def when_lineExtract_clicked(self, e):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        imageViewers = self.imageViewers

        if imageViewers and imageViewers[0].isEmpty :
            self.when_openBtn_clicked( e )
        else :
            pass
        pass
    pass # -- when_lineExtract_clicked

    def when_openBtn_clicked(self, e = None, fileName="" ):
        fun = inspect.getframeinfo(inspect.currentframe()).function
        log.info(fun)

        imageViewers = self.imageViewers

        if imageViewers and imageViewers[0]:
            fileName = imageViewers[0].loadImageFromFile(fileName=fileName, setFileName=True)

            _, ext = os.path.splitext(fileName)
            ext = ext.lower()

            if fileName:
                settings = self.settings
                recent_file_list = settings.value('recent_file_list', [], str)

                if fileName not in recent_file_list :
                    recent_file_list.insert( 0, fileName )

                    if len( recent_file_list ) > 9 :
                        recent_file_list.pop( len(recent_file_list) -1 )
                    pass

                    settings.setValue( "recent_file_list", recent_file_list )
                pass

                directory = os.path.dirname(fileName)
                log.info(f"dir = {directory}")

                find_files = f"{directory}/*{ext}"
                log.info(f"find_files={find_files}")

                import glob
                files = glob.glob(find_files)

                sel_file = None

                fileNameBase = os.path.basename(fileName)

                for file in files:
                    fileBase = os.path.basename(file)
                    log.info(f"fileBase = {fileBase}")

                    if fileBase > fileNameBase:
                        sel_file = file
                        break
                    pass
                pass

                log.info(f"sel_file = {sel_file}")

                if sel_file and imageViewers[1] :
                    imageViewers[1].loadImageFromFile(fileName=sel_file, setFileName=True)
                pass
            pass
        pass
    pass # -- when_openBtn_clicked

    def startEvent(self):
        settings = self.settings

        # 마지막 윈도우 크기 및 위치 로딩
        self.resize(self.settings.value("size", QSize(840, 640)))
        self.move(self.settings.value("pos", QPoint(50, 50)))
    pass # -- startEvent

    def showEvent(self, e ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        return QtWidgets.QMainWindow.showEvent(self, e)
    pass # -- showEvent

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

            zoomSlider = QtWidgets.QSlider( Qt.Horizontal )
            zoomSlider.setMinimum( 10 )
            zoomSlider.setMaximum( 400 )
            zoomSlider.setValue( 100 )

            zoomText = QtWidgets.QLineEdit( "100%" )
            zoomText.setReadOnly(1)
            zoomText.setFixedWidth(40)
            zoomText.setAlignment( Qt.AlignRight )

            fullExt = QtWidgets.QPushButton( "Full Extent")

            horizontal.addWidget( QtWidgets.QLabel( " Zoom ") )
            horizontal.addWidget( zoomSlider )
            horizontal.addWidget( zoomText )
            horizontal.addWidget( fullExt )
            spacer = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            horizontal.addItem( spacer )

            row += 1
            gridLayout.addLayout( horizontal, row, col )
        pass

        tab.setLayout( gridLayout )

    pass #-- init_tab

    def when_exitBtn_clicked(self, e):
        log.info( "when_exitBtn_clicked" )

        self.close()
    pass # -- when_exitBtn_clicked

    def closeEvent(self, e):
        log.info( "closeEvent")
        # 마지막 윈도우 크기 및 위치 저장 
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())

        e.accept()
    pass # -- closeEvent

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
    pass # -- show_image_on_image_viewer
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
    window = QtLineExtractor() # Create an instance of our class
    window.show()
    app.exec_() # Start the application
pass

