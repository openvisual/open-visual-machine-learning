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
class MyTableModel(QtCore.QAbstractTableModel):
    def __init__(self, tableView ):
        super(MyTableModel, self).__init__()

        self.tableView = tableView

        table_header = self.table_header()
        self.table_header = table_header

        self.col_count = len( table_header ) + 1

        self.dataList = self.create_data_list()  
    pass #-- __init__

    def remove_all_rows( self ) :
        self.dataList.clear()

        self.layoutChanged.emit()
    pass

    def appendData( self, data ):
        log.info( "appendData" )
        #self.dataList.append( data )

        self.dataList.append( data ) 

        row_count = len( self.dataList )

        col_count = self.col_count 
        x = col_count
        y = row_count

        self.dataChanged.emit(self.index(0, y), self.index( x, y)) 

        self.layoutChanged.emit()

        self.tableView.scrollToBottom() 
    pass # appendData

    def rowCount(self, index):
        # The length of the outer list.
        return len(self.dataList)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return self.col_count
    pass

    def format_of_value( self, index, value ) :
        col = index.column()

        fmt = ""

        tv = type( value )

        if tv == type( 0 ) :
            fmt = ",d"
        elif tv == type( 0.1 ) :
            if float( int( value ) ) == value :
                fmt = ",.0f"
            else : 
                fmt = ",.4f"
            pass
        pass

        0 and log.info( "type = %s, format = %s, value = %s" % (tv, fmt, value) )

        return fmt
    pass # -- format_of_value

    def data(self, index, role):
        row = index.row()
        col = index.column()

        if role == Qt.BackgroundRole :
            return self.getBackgroundBrush( index )
        elif role == Qt.ForegroundRole :
            return self.getForegroundBrush( index )
        else : 
            value = self.cell_value( index )

            if role == Qt.DisplayRole: 
                fmt = self.format_of_value( index, value )
                if fmt : 
                    value_org = value
                    value = format(value, fmt)

                    log.info( "row = %s, col = %s, fmt = %s, value = %s -> %s" % ( row, col, fmt, value_org, value ))
                    
                    return value
                else :
                    return value
                pass
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

    # headerData
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            table_header = self.table_header

            if col < len( table_header ) :
                header = table_header[ col ] 

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
    def getBackgroundBrush(self, index):
        row = index.row()
        
        if row%2 == 0 :
            return QBrush( Qt.lightGray )
        else : 
            return None
        pass
    pass # -- getBackgroundBrush
    
    # getForegroundBrush
    def getForegroundBrush(self, index):
        return None
    pass # -- getForegroundBrush

    def adjustColumnWidth( self ):
        tableView = self.tableView 
        
        # column width setting
        header = tableView.horizontalHeader()     
        
        col_count = self.col_count
        
        for idx in range( 0, col_count ) :
            if idx < col_count - 1 :
                header.setSectionResizeMode( idx, QHeaderView.ResizeToContents )
            else :
                header.setSectionResizeMode( idx, QHeaderView.Stretch )
            pass
        pass
        # -- column width setting
    pass # -- adjustColumnWidth

#TODO DatasetTableModel
pass #-- DatasetTableModel

class DatasetTableModel( MyTableModel ):
    def __init__(self, tableView ):
        super(DatasetTableModel, self).__init__( tableView ) 
    pass

    def table_header(self):
        table_header = [ "No.", "Question" , "Answer" ] 
        return table_header
    pass 

    def create_data_list(self) :
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
            questAns = self.dataList[ row ]
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
    
pass #-- DatasetTableModel

class LearnTableModel( MyTableModel ):
    def __init__(self, tableView ):
        super(LearnTableModel, self).__init__( tableView ) 
    pass

    def table_header(self):
        table_header = [ "No.", "Batch" , "Size", "Loss" , "Acc", "ETA", "Elapsed" ] 
        return table_header
    pass 

    def create_data_list(self) :
        dataList = []  

        return dataList
    pass #-- create_data 

    def cell_value( self, index) :
        row = index.row()
        col = index.column()

        0 and log.info( "row = %s, col = %s" % (row, col) )
        
        # cell value
        value = "" 

        if col == 0 :
            value = row + 1
        else :
            logs = self.dataList[ row ]

            table_header = self.table_header

            if col < len( table_header ) :  
                key = table_header[ col ]
                key = key.strip().lower()

                if key in logs :
                    value = logs[ key ]
                pass
            pass
        pass 

        log.info( "row = %s, col = %s, value = %s" % (row, col, value))

        return value
    pass # cell values
    
pass #-- LearnTableModel

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

        if 1 : 
            tableView = self.datasetTableView
            tableModel = DatasetTableModel( tableView )
            tableView.setModel( tableModel )

            tableModel.adjustColumnWidth()
        pass

        if 1 : 
            tableView = self.learnTableView
            tableModel = LearnTableModel( tableView )
            tableView.setModel( tableModel )

            tableModel.adjustColumnWidth()
        pass

        # connect signals to slots
        self.x.valueChanged.connect( self.when_x_valueChanged ) 
        self.x.editingFinished.connect( self.when_x_editingFinished ) 

        self.append.clicked.connect( self.when_append_clicked )
        self.start.clicked.connect( self.when_start_clicked )
        self.answer.clicked.connect( self.when_answer_clicked )
        self.myQuestion.textChanged.connect( self.when_my_question_textChanged )

        self.actionExit.triggered.connect(self.close_app)


        # 학습 모델 생성 
        model = keras.models.Sequential( )
        model.add(Dense(1, input_shape=[1] ))  
        model.compile( optimizer='adam', loss="mae", metrics=['accuracy'] )

        self.model = model

        self.statusbar.showMessage( "안녕하세요? 반갑습니다." )

        class Stream(QtCore.QObject):
            newText = QtCore.pyqtSignal(str)

            def write(self, text):
                self.newText.emit(str(text))
            pass

            def flush(self):
                pass
            pass
        pass

        sys.stdout = Stream(newText=self.onUpdateText)
    pass #MyQtApp __init__

    def close_app( self ):
        log.info( "close app" )
        self.hide()
        sys.exit()
    pass #-- close_app

    def when_my_question_textChanged( self, text ):
        log.info( "when_my_question_clicked" )

        answer = self.answer
        if len( text ) < 1 :
            answer.setDisabled( 1 )
        else :
            answer.setEnabled( 1 )
        pass
    pass # -- when_my_question_textChanged

    def when_answer_clicked( self ):
        log.info( "when_answer_clicked" )

        my_questions = [ 10 ]

        print( "\nMy Questions = " , my_questions )

        model = self.model

        my_answers = model.predict( my_questions ) 

        first_answer = my_answers[0][0]

        self.theAnswer.setText( "%s" % first_answer ) 
    pass # -- when_answer_clicked

    def onUpdateText(self, text):
        textEdit = self.learnState
        cursor = textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        textEdit.setTextCursor(cursor)
        textEdit.ensureCursorVisible()
    pass # -- onUpdateText

    # callbacks

    def on_train_begin(self, logs=None):
        print("on_train_begin" )

        self.curr_epoch = -1

        answer = self.answer
        myQuestion = self.myQuestion
        progressBar = self.progressBar
        
        progressBar.setValue( progressBar.minimum() )
        progressBar.setEnabled( True ) 

        answer.setDisabled( 1 )
        myQuestion.setDisabled( 1 )

        tableView = self.learnTableView 
        tableModel = tableView.model()
        tableModel.remove_all_rows() 

        self.statusbar.showMessage( "학습을 시작합니다." )
    pass # -- on_train_begin

    def on_train_end(self, logs=None):
        print("on_train_end" )

        self.curr_epoch = -2

        progressBar = self.progressBar
        myQuestion = self.myQuestion
        
        progressBar.setValue( progressBar.maximum() )
        progressBar.setDisabled( 1 ) 

        myQuestion.setEnabled( 1 ) 

        msg = "학습이 성공적으로 종료되었습니다. 질문을 입력하면 정답을 추론합니다."
        self.statusbar.showMessage( msg )
    pass # -- on_train_end

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        log.info("\n\nStart epoch {} of training; got log keys: {}\n".format(epoch, keys)) 

        self.epoch = epoch

        epochs = self.epochs
        value = (100*epoch)/epochs
        
        progressBar = self.progressBar
        progressBar.setValue( int( value ) ) 
        self.statusbar.showMessage( "학습 %d 단계가 진행중입니다." % epoch )

        tableView = self.learnTableView
        tableModel = tableView.model()

        tableModel.appendData( logs ) 

    pass # -- on_epoch_begin

    def on_epoch_end(self, epoch, logs=None):
        #  ['loss', 'mean_squared_error', 'val_loss', 'val_mean_squared_error']
        keys = list(logs.keys())

        #log.info( "%s" % keys )
        
        loss = logs[ "loss" ] #TODO 2010 손실 값 키 설정 loss key set 

        log.info( "\ncurr epoch[%s] val_loss = %s\n" % ( epoch + 1, loss ) ) 

        log.info("\nEnd epoch {} of training; got log keys: {}\n".format(epoch, keys)) 

        if 1 : 
            # update learn table
            tableView = self.learnTableView
            tableModel = tableView.model()
            row = epoch
            row_data = tableModel.dataList[ row ]
            row_data[ "loss" ] = logs[ "loss" ]
            row_data[ "acc" ]  = logs[ "acc" ]
            col_count = tableModel.col_count 

            x = col_count
            y = row

            #tableModel.dataChanged.emit(tableModel.index(0, y), tableModel.index( x, y))  
            tableModel.layoutChanged.emit()
        pass

        epochs = self.epochs 
        value = (100*epoch)/epochs 
        progressBar = self.progressBar 

        progressBar.setValue( int( value ) ) 
    pass #-- on_epoch_end

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        0 and log.info("...Training: start of batch {}; got log keys: {}".format(batch, keys))

        size = logs[ "size" ]

        if 1 : 
            # update learn table
            tableView = self.learnTableView
            tableModel = tableView.model()
            row = self.epoch
            row_data = tableModel.dataList[ row ]
            row_data[ "size" ] = logs[ "size" ]
            col_count = tableModel.col_count 

            x = col_count
            y = row

            #tableModel.dataChanged.emit(tableModel.index(0, y), tableModel.index( x, y))  
            tableModel.adjustColumnWidth()
            tableModel.layoutChanged.emit()
        pass
    pass # -- on_train_batch_begin

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        0 and log.info("...Training: end of batch {}; got log keys: {}".format(batch, keys))

        size = logs[ "size" ]
        loss = logs[ "loss" ]
        acc  = logs[ "acc"  ]

        if 1 : 
            # update learn table
            tableView = self.learnTableView
            tableModel = tableView.model()
            row = self.epoch
            row_data = tableModel.dataList[ row ]
            row_data[ "loss" ] = logs[ "loss" ]
            row_data[ "acc" ]  = logs[ "acc" ]
            row_data[ "size" ] = logs[ "size" ]
            col_count = tableModel.col_count 

            x = col_count
            y = row

            #tableModel.dataChanged.emit(tableModel.index(0, y), tableModel.index( x, y))  
            tableModel.layoutChanged.emit()
        pass
    pass # -- on_train_batch_begin

    # -- callbacks

    def when_start_clicked( self ) : 
        log.info( "when_start_clicked" )

        self.learnState.setText( "" )

        tableView = self.datasetTableView
        tableModel = tableView.model()

        questAnsList = tableModel.dataList

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

        self.statusbar.showMessage( "데이터셋에 질문과 정답이 추가되었습니다." )
    pass #-- when_append_clicked

    def when_x_editingFinished(self) :
        log.info( "when_x_editingFinished" )

        #self.appendData()
    pass #-- when_x_editingFinished

    def appendData( self ) : 
        tableView = self.datasetTableView 
        tableModel = tableView.model()
        
        x = self.x.value() 
        y = int( self.y.text().strip() )
        questAns = QuestAns( x, y )

        tableModel.appendData( questAns )
    pass #-- when_x_editingFinished 
pass # -- MyQtApp

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
    app.setWindowIcon(QtGui.QIcon('window_icon_01.png')) 
    window = MyQtApp() # Create an instance of our class
    window.show()
    window.setWindowIcon(QtGui.QIcon('window_icon_01.png')) 
    app.exec_() # Start the application
pass # -- main