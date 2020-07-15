# coding: utf-8

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QPushButton

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self ):
        super(TableModel, self).__init__()

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
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self.data[index.row()][index.column()]
        pass
    pass

    def rowCount(self, index):
        # The length of the outer list.
        return len(self.data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self.data[0])
    pass
pass

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.table = QtWidgets.QTableView()

        self.model = TableModel()

        self.table.setModel(self.model)

        self.setCentralWidget(self.table)

        btn = QPushButton("button")

        btn.clicked.connect( self.when_btn_clicked )

        self.table.setIndexWidget( self.model.index(0, 2), btn)  
    pass

    def when_btn_clicked( self ) : 
        print( "when_btn_clicked" )
        model = self.model
        data = model.data
        model.data.append( [17, 18, 19] )
        model.layoutChanged.emit()
    pass
pass

app=QtWidgets.QApplication(sys.argv)
window=MainWindow()
window.show()
app.exec_()

# end