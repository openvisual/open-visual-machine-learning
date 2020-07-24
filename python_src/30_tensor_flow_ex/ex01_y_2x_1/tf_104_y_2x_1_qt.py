# -*- coding: utf-8 -*-

import os, sys, datetime 

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QPushButton 
from PyQt5.Qt import *

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import callbacks 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda  
from tensorflow.python.keras import backend as K 

class QuestAns :
    def __init__(self, quest, answer) :
        self.quest = quest
        self.answer = answer
    pass
pass

#TODO      DatasetTableModel
class DatasetTableModel(QtCore.QAbstractTableModel):
    def __init__(self, tableView ):
        super(DatasetTableModel, self).__init__()

        self.colCount = 4 

        self.tableView = tableView

        self.table_header = [ "Question" , "Answer" ] 

        self.data = self.create_data()  
    pass #-- __init__

    def appendData( self, data ):
        log.info( "appendData" )
        #self.data.append( data )

        self.data.append( data ) 

        y = len( self.data )

        self.dataChanged.emit(self.index(0, y), self.index( self.colCount , y)) 

        self.layoutChanged.emit()

        self.tableView.scrollToBottom()
    pass # appendData

    def create_data(self) :
        questAnsList = []

        # 질문/정답 만들기 
        questAnsList.append( QuestAns( -1.0, -3.0 ) )
        questAnsList.append( QuestAns( 0.0, -1.0 ) )
        questAnsList.append( QuestAns( 1.0, 1.0 ) )
        questAnsList.append( QuestAns( 2.0, 1.0 ) )
        questAnsList.append( QuestAns( 3.0, 5.0 ) )
        questAnsList.append( QuestAns( 4.0, 7.0 ) )
        questAnsList.append( QuestAns( 5.0, 9.0 ) )

        return questAnsList
    pass #-- create_data

    def cell_value( self, index) :
        row = index.row()
        col = index.column()

        # get cell value
        value = "" 

        if col == 0 :
            value = row + 1
        else :
            questAns = self.data[ row ]
            if col == 1 :
                value = questAns.quest 
            elif col == 2 :
                value = questAns.answer
            else : 
                value = ""
            pass
        pass 

        return value
    pass # cell values    

    def data(self, index, role):
        if role == Qt.BackgroundRole :
            return self.getBackgroundBrush( index )
        elif role == Qt.ForegroundRole :
            return self.getForegroundBrush( index )
        else : 
            value = self.cell_value( index )

            if role == Qt.DisplayRole: 
                return value
            elif Qt.TextAlignmentRole == role : 
                if type( value ) == int :
                    return  Qt.AlignRight | QtCore.Qt.AlignVCenter
                elif type( value ) == float :
                    return  Qt.AlignRight | QtCore.Qt.AlignVCenter
                elif isinstance( value, datetime.date) :
                    return Qt.AlignHCenter | QtCore.Qt.AlignVCenter
                else :
                    return Qt.AlignLeft | QtCore.Qt.AlignVCenter
                pass
            pass # -- return cell alignment value
        pass
    pass #-- data

    def rowCount(self, index):
        # The length of the outer list.
        return len(self.data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return self.colCount
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
        
        if row%2 == 0 :
            return QBrush( Qt.lightGray )
        else : 
            return None
        pass
    pass # -- getBackgroundBrush
    
    # getForegroundBrush
    def getForegroundBrush(self , index ):
        return None
    pass # -- getForegroundBrush

    def adjustColumnWidth( self ):
        tableView = self.tableView 
        
        # column width setting
        header = tableView.horizontalHeader()     
        
        colCount = self.columnCount( tableView )
        
        for idx in range( 0, colCount ) :
            if colCount - 1 > idx :
                header.setSectionResizeMode( idx, QHeaderView.ResizeToContents )
            else :
                header.setSectionResizeMode( idx, QHeaderView.Stretch )
            pass
        pass
        # -- column width setting
    pass # -- adjustColumnWidth

#TODO DatasetTableModel
pass #-- DatasetTableModel

class Stream(QtCore.QObject):
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))
    pass

    def flush(self):
        pass
    pass
pass

class MyQtApp(QtWidgets.QMainWindow, callbacks.Callback):

    def __init__(self):
        super(MyQtApp, self).__init__() # Call the inherited classes __init__ method
        
        uic.loadUi( './tf_104_y_2x_1_qt.ui', self) # Load the .ui file 

        progressBar = self.progressBar
        
        progressBar.setValue( 0 )
        progressBar.setDisabled( 1 )

        answer = self.answer
        myQuestion = self.myQuestion
        
        answer.setDisabled( 1 )
        myQuestion.setDisabled( 1 )
        myQuestion.setValidator( QtGui.QDoubleValidator() )

        tableView = self.datasetTableView
        tableModel = DatasetTableModel( tableView )
        tableView.setModel( tableModel )

        tableModel.adjustColumnWidth()

        # connect signals to slots
        self.x.valueChanged.connect( self.when_x_valueChanged ) 
        self.x.editingFinished.connect( self.when_x_editingFinished ) 

        self.append.clicked.connect( self.when_append_clicked )
        self.start.clicked.connect( self.when_start_clicked )
        self.answer.clicked.connect( self.when_answer_clicked )

        # 학습 모델 생성 
        model = keras.models.Sequential( )
        model.add(Dense(1, input_shape=[1] ))  
        model.compile( optimizer='adam', loss="mae", metrics=['accuracy'] )

        self.model = model

        sys.stdout = Stream(newText=self.onUpdateText)
    pass #MyQtApp __init__

    def when_answer_clicked( self ):
        log.info( "when_answer_clicked" )

        my_questions = [ 10 ]

        print( "\nMy Questions = " , my_questions )

        model = self.model

        my_answers = model.predict( my_questions ) 

        first_answer = my_answers[0][0]

        self.theAnswer.setText( "%s" % first_answer ) 
    pass

    def onUpdateText(self, text):
        textEdit = self.learningState
        cursor = textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        textEdit.setTextCursor(cursor)
        textEdit.ensureCursorVisible()
    pass

    # callbacks

    def on_train_begin(self, logs=None):
        print("on_train_begin" )

        progressBar = self.progressBar
        
        progressBar.setValue( progressBar.minimum() )
        progressBar.setEnabled( True ) 
    pass

    def on_train_end(self, logs=None):
        print("on_train_end" )

        progressBar = self.progressBar
        
        progressBar.setValue( progressBar.maximum() )
        progressBar.setDisabled( True ) 

        myQuestion = self.myQuestion
        myQuestion.setDisabled( False )
        
        answer = self.answer
        answer.setDisabled( False )
    pass

    def on_epoch_begin(self, epoch, logs=None):
        log.info("\n\nStart epoch %s of training." % ( epoch  + 1 ) )

        epochs = self.epochs

        value = (100*epoch)/epochs
        
        progressBar = self.progressBar
        
        progressBar.setValue( int( value ) ) 
    pass

    def on_epoch_end(self, epoch, logs=None):
        #  ['loss', 'mean_squared_error', 'val_loss', 'val_mean_squared_error']
        keys = list(logs.keys())

        #log.info( "%s" % keys )
        
        loss = logs[ "loss" ] #TODO 2010 손실 값 키 설정 loss key set 

        log.info( "\ncurr epoch[%s] val_loss = %s\n" % ( epoch + 1, loss ) ) 

        log.info("\nEnd epoch {} of training; got log keys: {}\n".format(epoch, keys)) 

        epochs = self.epochs

        value = (100*epoch)/epochs
        
        progressBar = self.progressBar
        
        progressBar.setValue( int( value ) ) 
    pass

    # -- callbacks

    def when_start_clicked( self ) : 
        log.info( "when_start_clicked" )

        self.learningState.setText( "" )

        tableView = self.datasetTableView
        tableModel = tableView.model()

        questAnsList = tableModel.data 

        questions = [ qa.quest for qa in questAnsList ]
        answers = [ qa.answer for qa in questAnsList ]

        print( "questions: ", questions )
        print( "answers: ", answers )

        questions = questions*100
        answers = answers*100

        model = self.model

        self.epochs = 30

        epochs = self.epochs

        callbacks = [ self ]
        model.fit( questions, answers, epochs=epochs, batch_size=7, callbacks=callbacks )  

    pass #-- when_start_clicked

    def when_x_valueChanged(self, i ) :
        log.info( "when_x_changed: %s" % i )

        y =  2*i - 1
        self.y.setText( "%s" % ( y ) )
    pass #-- when_x_valueChanged

    def when_append_clicked( self ):
        log.info( "when_append_clicked" )

        self.appendData()
    pass #-- when_append_clicked

    def when_x_editingFinished(self) :
        log.info( "when_x_editingFinished" )

        self.appendData()
    pass #-- when_x_editingFinished

    def appendData( self ) : 
        tableView = self.datasetTableView 
        tableModel = tableView.model()
        
        x = self.x.value() 
        y = int( self.y.text().strip() )
        questAns = QuestAns( x, y )

        tableModel.appendData( questAns )
    pass #-- when_x_editingFinished 
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

