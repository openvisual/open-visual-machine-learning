# -*- coding: utf-8 -*-
import os, sys, datetime, time, numpy as np , matplotlib
from time import sleep

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QLineEdit
from PyQt5.Qt import *

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.python.keras import backend as K

matplotlib.use('QT5Agg')

import matplotlib.pylab as pylab
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class QuestAns :
    def __init__(self, quest, answer) :
        self.quest = quest
        self.answer = answer
    pass
pass

# TODO      DatasetTableModel
class MyTableModel(QtCore.QAbstractTableModel):
    def __init__(self, tableView ):
        super(MyTableModel, self).__init__()

        self.tableView = tableView

        table_header = self.table_header()
        self.table_header = table_header

        self.col_count = len( table_header ) + 1

        self.dataList = self.create_data_list()

        self.dataChanged.connect(self.repaintTableView)
    pass # -- __init__

    def repaintTableView(self):
        log.info( "repaintTableView()")
    pass

    def remove_all_rows( self ) :
        self.dataList.clear()

        self.layoutChanged.emit()
    pass # -- remove_all_rows

    def appendData( self, data ):
        log.info( "appendData" )

        #self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount() + 1)

        self.dataList.append( data )

        #self.endInsertRows()

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

        if not index.isValid():
            return QVariant()
        elif role == Qt.BackgroundRole :
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

                    0 and log.info( "row = %s, col = %s, fmt = %s, value = %s -> %s" % ( row, col, fmt, value_org, value ))

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
            else :
                return QVariant()
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

        for idx in range( col_count ) :
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
        table_header = [ "No.", "Loss" , "Acc", "Count", "Elapsed", "ETA" ]
        return table_header
    pass

    def create_data_list(self) :
        dataList = []

        if 0 :
            logs = { "acc" : 0.2857143, "loss" : 0.571787 }
            dataList.append( logs )
        pass

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

            table_header_len = len( table_header )

            if col < table_header_len :
                key = table_header[ col ]
                key = key.strip().lower()

                if key in logs :
                    value = logs[ key ]

                    0 and log.info( "row = %s, col = %s, key = %s, value = %s" % (row, col, key, value))
                else :
                    value = "__"
                pass 
            else :
                value = ""
            pass
        pass

        0 and log.info( "row = %s, col = %s, value = %s" % (row, col, value))

        return value
    pass # cell values

pass # -- LearnTableModel

class MyQtApp(QtWidgets.QMainWindow, callbacks.Callback):

    def __init__(self):
        super(MyQtApp, self).__init__() # Call the inherited classes __init__ method

        self.epochs = 30

        uic.loadUi( './tf_104_y_2x_1_qt.ui', self) # Load the .ui file

        # central widget layout margin
        self.centralWidget().layout().setContentsMargins( 9, 9, 9, 9 )

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

        self.init_plot_content()

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
    pass # MyQtApp __init__

    def init_plot_content(self):
        # TODO init plot content
        # plot graph
        # self.plotWidget = pg.PlotWidget( title="Loss/Accuracy" )
        fig, ax = pylab.subplots()

        self.fig = fig
        self.ax = ax

        epochs = self.epochs
        if 1:
            x = [0, epochs]
            y = [0, 0]
            ax.plot(x, y, label="acc")
        pass

        if 1:
            x = [0]
            y = [1]
            ax.plot(x, y, label="loss", linewidth=1)
        pass

        ax.legend(loc="upper right")

        self.figureCanvas = FigureCanvas(fig)
        self.lines = []

        figureCanvas = self.figureCanvas

        layout = QtWidgets.QVBoxLayout(self.plot_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(figureCanvas)
    pass  # -- init_plot_content

    def close_app( self ):
        log.info( "close app" )
        self.hide()
        sys.exit()
    pass # -- close_app

    def when_my_question_textChanged( self, text ):
        log.info( "when_my_question_clicked" )

        text_org = text

        text = text.replace( ",", "" ).strip()

        ts = text.split( "." )

        text = "{:,d}".format( int( ts[0] ) )

        if len( ts ) == 2 :
            text += "."
            if len( ts[1] ) > 0 :
                text += ts[1]
            pass
        pass

        if text_org != text :
            self.myQuestion.setText( text )
        pass

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

    def update_row_data_from_logs(self, row_data, logs):

        idx = 0

        if ("acc" in logs) or ("accuracy" in logs) :
            acc = 0.0
            if "acc" in logs :
                acc = logs[ "acc" ]
            else :
                acc = logs[ "accuracy"]
            pass

            acc = 0.0 + float( acc )
            row_data[ "acc" ] = acc

            log.info("[%02d] row_data[%s] = %s" % (idx, "acc", acc ) )

            idx += 1
        pass

        if "loss" in logs :
            loss = logs[ "loss" ]
            loss = 0.0 + float( loss )
            row_data[ "loss" ] = loss

            log.info("[%02d] row_data[%s] = %s" % (idx, "loss", loss ) )

            idx += 1
        pass

        # 경과 시간 계산
        row_data["elapsed"] = time.time() - row_data["elapsed_then"]

    pass  # -- update_row_data_from_logs

    # callbacks

    def on_train_begin(self, logs=None):
        # TODO 훈련 시작
        print("on_train_begin" )

        self.epoch = -1

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

        if 1 : # init lines
            ax = self.ax
            fig = self.fig

            ax.clear()

            self.ax.set_autoscaley_on(True)

            self.acc_list = []
            self.loss_list = []
            self.lines = []

            x = []
            acc_list = self.acc_list
            loss_list = self.loss_list
            lines = self.lines

            lines.append( ax.plot( x, acc_list , label="acc") )
            lines.append( ax.plot( x, loss_list, label="loss" ) )

            ax.legend( loc="upper right" )

            fig.canvas.draw()
            #fig.canvas.flush_events()
        pass # -- init lines

        self.statusbar.showMessage( "학습을 시작합니다." )
    pass # -- on_train_begin

    def on_train_end(self, logs=None):
        print("on_train_end" )

        self.epoch = -2

        progressBar = self.progressBar
        myQuestion = self.myQuestion

        progressBar.setValue( progressBar.maximum() )
        progressBar.setDisabled( 1 )

        myQuestion.setEnabled( 1 )

        msg = "학습이 성공적으로 종료되었습니다. 질문을 입력하면 정답을 추론합니다."
        self.statusbar.showMessage( msg )

        if 1 :
            # update learn table
            tableView = self.learnTableView
            tableModel = tableView.model()
            dataList = tableModel.dataList
        pass
    pass # -- on_train_end

    def on_epoch_begin(self, epoch, logs=None):
        log.info("on_epoch_begin {} of training; got log keys: {}\n".format(epoch, list( logs.keys() )))

        self.epoch = epoch

        epochs = self.epochs
        value = (100*epoch)/epochs

        progressBar = self.progressBar
        progressBar.setValue( int( value ) )
        self.statusbar.showMessage( "학습 %d 단계가 진행중입니다." % epoch )

        tableView = self.learnTableView
        tableModel = tableView.model()

        row_data = logs.copy()
        row_data[ "count" ] = 0

        row_data[ "elapsed_then" ] = time.time()
        row_data[ "elapsed" ] = 0

        tableModel.appendData( row_data )

    pass # -- on_epoch_begin

    def on_epoch_end(self, epoch, logs=None):
        log.info("on_epoch_end {} of training; got log keys: {}".format(epoch, list(logs.keys())))

        # update learn table
        tableView = self.learnTableView
        tableModel = tableView.model()
        row = epoch
        dataList = tableModel.dataList
        row_data = dataList[ row ]

        self.update_row_data_from_logs( row_data, logs)

        tableModel.layoutChanged.emit()

        if 1 :  # plot chart
            ax = self.ax
            fig = self.fig

            lines = self.lines
            epochs = self.epochs

            acc_list = self.acc_list
            loss_list = self.loss_list

            acc = 0
            if "acc" in logs :
                acc = logs[ "acc" ]
            elif "accuracy" in logs :
                acc = logs["accuracy"]
            pass

            loss = logs[ "loss" ]

            0 and log.info( f"acc = {acc}, loss = {loss}")

            acc = 0.0 + float( acc )
            loss = 0.0 + float(loss)

            acc_list.append( acc )
            loss_list.append( loss )

            x_data = [i for i, _ in enumerate(acc_list)]

            lines[0][0].set_data(x_data, acc_list)
            lines[1][0].set_data(x_data, loss_list)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()

            log.info("Done. Line plotting")
        pass  # -- plot chart

        epochs = self.epochs
        value = (100*epoch)/epochs
        progressBar = self.progressBar

        progressBar.setValue( int( value ) )
    pass #-- on_epoch_end

    def on_train_batch_begin(self, batch, logs=None):
        1 and log.info("on_train_batch_begin {}; got log keys: {}".format(batch, list(logs.keys())))

        if 1 :
            # update learn table
            tableView = self.learnTableView
            tableModel = tableView.model()
            row = self.epoch
            row_data = tableModel.dataList[ row ]

            row_data["count"] += 1

            self.update_row_data_from_logs( row_data, logs )

            col_count = tableModel.col_count

            x = col_count
            y = row

            #tableModel.dataChanged.emit(tableModel.index(0, y), tableModel.index( x, y))
            #tableModel.adjustColumnWidth()
            tableModel.layoutChanged.emit()
        pass
    pass # -- on_train_batch_begin

    def on_train_batch_end(self, batch, logs=None):
        1 and log.info("on_train_batch_end {}; got log keys: {}".format(batch, list(logs.keys())))

        if 1 :
            # update learn table
            tableView = self.learnTableView
            tableModel = tableView.model()
            row = self.epoch
            row_data = tableModel.dataList[ row ]

            self.update_row_data_from_logs( row_data, logs)

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

        epochs = self.epochs

        callbacks = [ self ]
        self.hist = model.fit( questions, answers, epochs=epochs, batch_size=7, callbacks=callbacks )

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
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    log.info( "Pwd 1: %s" % os.getcwd())
    # change working dir to current file
    dirname = os.path.dirname(__file__)
    if dirname :
        log.info( "dirname: %s" % dirname )
        os.chdir(dirname)
        log.info( "Pwd 2: %s" % os.getcwd())
    pass

    # Create an instance of QtWidgets.QApplication
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('window_icon_01.png'))
    window = MyQtApp() # Create an instance of our class
    window.show()
    window.setWindowIcon(QtGui.QIcon('window_icon_01.png'))
    app.exec_() # Start the application
pass # -- main