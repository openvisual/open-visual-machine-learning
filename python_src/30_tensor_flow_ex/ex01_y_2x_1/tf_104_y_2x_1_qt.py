# -*- coding: utf-8 -*-

import os, sys
from PyQt5 import QtWidgets, uic 

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QPushButton 
from PyQt5.Qt import *

#TODO      DatasetTableModel
class DatasetTableModel(QtCore.QAbstractTableModel):
    def __init__(self ):
        super(DatasetTableModel, self).__init__()

        self.table_header = [ "y" , "x" ]

        data = [
            [4, 9, 2],
            [1, 0, 0],
            [3, 5, 0],
            [3, 3, 2],
            [7, 8, 9],
        ]
        
        self.data = data
    pass

    def data(self, index, role):
        if role == Qt.BackgroundRole :
            return self.getBackgroundBrush( index )
        elif role == Qt.ForegroundRole :
            return self.getForegroundBrush( index )
        elif role == Qt.DisplayRole: 
            row = index.row()
            col = index.column()

            if col == 0 :
                return row + 1
            else :
                return self.data[ row ][ col - 1 ]
            pass
        pass
    pass

    def rowCount(self, index):
        # The length of the outer list.
        return len(self.data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self.data[0]) + 1
    pass

    # headerData
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            table_header = self.table_header

            if col == 0 :
                return "No."
            elif col <= len( table_header ) :
                header = table_header[ col - 1 ] 

                return header
            else :
                return ""
            pass

        else :
            return None
        pass
    pass
    # -- headerData
    # getBackgroundBrush
    def getBackgroundBrush(self , index ):
        row = index.row()
        
        if row%2 == 1 :
            return QBrush( Qt.lightGray )
        else : 
            return None
        pass
    pass
    # -- getBackgroundBrush
    
    # getForegroundBrush
    def getForegroundBrush(self , index ):
        return None
    pass
    # -- getForegroundBrush

#TODO DatasetTableModel
pass #-- DatasetTableModel

class MyQtApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__() # Call the inherited classes __init__ method
        
        uic.loadUi( './tf_104_y_2x_1_qt.ui', self) # Load the .ui file 

        progressBar = self.progressBar
        
        progressBar.setValue( 0 )
        progressBar.setDisabled( True )

        datasetTableView = self.datasetTableView
        datasetTableModel = DatasetTableModel()
        datasetTableView.setModel( datasetTableModel )
    pass

    def when_MyPushButton_clicked(self) :
        print( "when_MyPushButton_clicked" )

        myText = self.myLineEdit.text()
        self.myLabel.setText( myText )
    pass
pass

if __name__ == '__main__':
    print( "Pwd 1: %s" % os.getcwd())
    # change working dir to current file
    dirname = os.path.dirname(__file__)
    if dirname : 
        print( "dirname: ", dirname )
        os.chdir(dirname)
        print( "Pwd 2: %s" % os.getcwd())
    pass

    # Create an instance of QtWidgets.QApplication
    app = QtWidgets.QApplication(sys.argv)
    window = MyQtApp() # Create an instance of our class
    window.show()
    app.exec_() # Start the application
pass

