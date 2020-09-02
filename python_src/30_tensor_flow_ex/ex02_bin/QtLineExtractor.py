# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, sys, inspect
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtWidgets import QApplication, QWidget, QAction
from PyQt5.QtCore import QSettings, QPoint, QSize, Qt, QModelIndex
from PyQt5.QtGui import QStandardItemModel, QStandardItem

from rsc.my_qt import *
from QtImageViewer import *

from Common import *
from LineExtractor import *

class QtLineExtractor(QtWidgets.QMainWindow, Common ):

    def __init__(self):
        QtWidgets.QMainWindow.__init__( self )
        Common.__init__(self)

        uic.loadUi('./QtLineExtractor.ui', self)

        self.settings = QSettings('TheOneTech', 'line_extractor')

        self.imageViewers = []

        self.init_tab( self.tabWidgetLeft )
        self.init_tab( self.tabWidgetRight )

        self.progressBar.setValue( 0 )
        self.progressBar.setEnabled( 0 )

        self.durationLcdNumber.display( QtCore.QTime.currentTime().toString('hh:mm:ss') )

        self.lineMatchComboBox.addItems( [ "Matched", "All", "A Only", "B Only"] )

        if 1 :
            headerLabels = [ '그림', 'ID', '유사도', '길이', '좌표(A)', '좌표(B)' ]
            colLen = len(headerLabels)

            model = QStandardItemModel()
            model.setHorizontalHeaderLabels(headerLabels)

            tableView = self.lineTableView
            tableView.setModel(model)

            header = tableView.horizontalHeader()
            header.setDefaultAlignment(Qt.AlignHCenter)

            for i in range( colLen ) :
                #header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
                header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
            pass

            # Sets different alignment data just on the first column
            #model.setHeaderData(0, Qt.Horizontal, Qt.AlignJustify, Qt.TextAlignmentRole )
            #model.setHeaderData(0, Qt.Horizontal, Qt.AlignJustify, Qt.TextAlignmentRole)

            for row in range( 4 ):
                items = []
                for col in range( colLen ):
                    data = f"{(row + 1) * (col + 1)}"

                    item = QStandardItem( data )
                    item.setEditable( False )

                    items.append( item )
                pass

                model.insertRow( row, items )
            pass

            tableView.setWindowTitle( "Lines Extracted" )
            tableView.resizeColumnsToContents()
        pass

        # signal -> slot connect
        self.tabWidgetLeft.currentChanged.connect( self.when_tab_widget_current_changed )
        self.exitBtn.clicked.connect( self.when_exitBtn_clicked )
        self.actionExit.triggered.connect(self.close_app)
        self.actionOpen.triggered.connect( self.when_openBtn_clicked )
        self.openBtn.clicked.connect( self.when_openBtn_clicked )
        self.lineExtract.clicked.connect( self.when_lineExtract_clicked )
        self.actionFull_Screen.triggered.connect( self.when_fullScreen_clicked )
        self.prevFileOpen.clicked.connect( self.when_prevFileOpen_clicked )
        self.nextFileOpen.clicked.connect(self.when_nextFileOpen_clicked)
        # -- signal -> slot connect

        self.buildOpenRecentFilesMenuBar()
        self.startEvent()

        self.paintUi()
    pass # -- __init__

    def plot_image(self, image, mode="A", title="", border_color="black" ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)
    pass

    def plot_histogram(self, image, mode="A"):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # 히스토 그램 표출
        pass
    pass

    def paintUi(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        is_file_open = False

        prevFile = None
        nextFile = None

        imageViewers = self.imageViewers

        if imageViewers and not imageViewers[0].isEmpty :
            is_file_open = True

            fileName = imageViewers[0].fileName
            if fileName :
                prevFile = self.prev_file( fileName )
                nextFile = self.next_file( fileName )
            pass
        pass

        log.info( f"prevFile={prevFile}")
        log.info( f"nextFile={nextFile}")

        self.prevFileOpen.setEnabled( prevFile is not None )
        self.nextFileOpen.setEnabled( nextFile is not None )
        self.lineExtract.setEnabled( is_file_open )
        self.viewJson.setEnabled(is_file_open)
    pass # -- paintUi

    def when_nextFileOpen_clicked(self, e):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        imageViewers = self.imageViewers

        if imageViewers :
            imageViewer = imageViewers[0]
            if imageViewer is not None and not imageViewer.isEmpty and imageViewer.fileName :
                fileName = imageViewer.fileName
                nextFile = self.next_file( fileName )
                log.info( f"nextFile = {nextFile}" )
                if nextFile :
                    self.when_openBtn_clicked( fileName = nextFile )
                    self.statusBar().showMessage( f"Next File [ {nextFile} ] is opened.")
                pass
            pass
        pass

    pass # -- when_nextFileOpen_clicked

    def when_prevFileOpen_clicked(self, e):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        imageViewers = self.imageViewers

        if imageViewers :
            imageViewer = imageViewers[0]
            if imageViewer is not None and not imageViewer.isEmpty and imageViewer.fileName :
                fileName = imageViewer.fileName
                prevFile = self.prev_file( fileName )
                log.info( f"nextFile = {prevFile}" )
                if prevFile :
                    self.when_openBtn_clicked( fileName = prevFile )
                    self.statusBar().showMessage( f"Previous File [ {prevFile} ] is opened.")
                pass
            pass
        pass

    pass # -- when_prevFileOpen_clicked

    def resizeEvent(self, event):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        action = self.actionFull_Screen

        state = self.windowState()

        if state == Qt.WindowFullScreen :
            action.setText( "Exit full screen" )
        else :
            action.setText( "Full Screen" )
        pass
    pass # -- resizeEvent

    def keyPressEvent(self, e):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        state = self.windowState()

        if e.key() == QtCore.Qt.Key_Escape and state == Qt.WindowFullScreen :
            self.showNormal()
        elif e.key() == QtCore.Qt.Key_F11:
            if self.isFullScreen() :
                self.showNormal()
            else:
                self.showFullScreen()
            pass
        pass
    pass # -- keyPressEvent

    def when_fullScreen_clicked(self, e):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        if self.isFullScreen() :
            self.showNormal()
        else :
            self.showFullScreen()
        pass

    pass # -- when_fullScreen_clicked

    def buildOpenRecentFilesMenuBar(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        menuOpen_Recent = self.menuOpen_Recent

        settings = self.settings

        recent_file_list = settings.value( 'recent_file_list', [], str)

        menuOpen_Recent.setEnabled( len( recent_file_list ) > 0 )

        menuOpen_Recent.clear()

        for i, fileName in enumerate( recent_file_list ):
            log.info( f"fileName = {fileName}" )
            action = QAction( f"[{(i+1):d}] {fileName}", menuOpen_Recent )
            action.fileName = fileName

            if i == 0 :
                action.setShortcut('Ctrl+R')
            pass

            menuOpen_Recent.addAction( action )

            action.triggered.connect( lambda val : self.when_recentFileAction_clicked( f"{fileName}" ) )
        pass

        if len( recent_file_list ) :
            menuOpen_Recent.addSeparator()

            action = QAction("Clear Recently Opened")
            menuOpen_Recent.addAction(action)

            action.triggered.connect(self.when_clear_recently_opened_clicked)
            action.setEnabled(len(recent_file_list) > 0)
        pass

    pass # -- buildOpenRecentFilesMenuBar

    def when_clear_recently_opened_clicked(self, e):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        settings = self.settings
        settings.setValue("recent_file_list", [])

        self.buildOpenRecentFilesMenuBar()
    pass

    def when_recentFileAction_clicked(self, fileName ):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        self.when_openBtn_clicked( e=None, fileName=fileName )
    pass # -- when_recentFileAction_clicked

    def when_lineExtract_clicked(self, e):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        imageViewers = self.imageViewers

        for i, imageViewer in enumerate( imageViewers ):
            lineExtractor = LineExtractor()

            img_path = imageViewer.fileName

            mode = chr( ord( 'A') + i )

            log.info( f"img_path={img_path}, mode={mode}" )

            lineExtractor.my_line_extract(img_path=img_path, qtUi=self, mode=mode)

            if i == len( imageViewers ) - 1 :
                # 결과창 폴더 열기
                folder = "c:/temp"
                lineExtractor.open_file_or_folder(folder)
            pass
        pass

    pass # -- when_lineExtract_clicked

    def when_openBtn_clicked(self, e = None, fileName="" ):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )
        debug = False

        imageViewers = self.imageViewers

        if imageViewers and imageViewers[0]:
            fileName = imageViewers[0].loadImageFromFile(fileName=fileName, setFileName=True)

            self.statusBar().showMessage(f"File [ {fileName} ] is opened.")

            if fileName:
                self.save_recent_file(self.settings, fileName)
                self.buildOpenRecentFilesMenuBar()

                file_next = self.next_file( fileName )

                if file_next and imageViewers[1] :
                    imageViewers[1].loadImageFromFile(fileName=file_next, setFileName=True)
                pass
            pass
        pass

        self.paintUi()
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

