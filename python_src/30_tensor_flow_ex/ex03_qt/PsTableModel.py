# -*- coding: utf-8 -*-

import inspect
import datetime
import time

from qgis.PyQt import QtCore

from PyQt5.QtWidgets import QTableView, QHeaderView, QApplication
from PyQt5.QtCore import Qt , QAbstractTableModel 
from psCom.PsCom import log
from abc import abstractmethod 
from PyQt5.Qt import QBrush
from psModel.PsCol import PsCol
from psCom.PsComWidget import PsComWidget

# ErpTableModel
class PsTableModel( QAbstractTableModel , PsComWidget ): 
    
    debug = True 
    
    __abstract__ = True 
    
    def __init__(self, parent, tableView ):
        QAbstractTableModel.__init__(self, parent )
        
        self.window = parent 
        
        self.order_by_idx = None 
        
        self.useSelector = False 
        
        self.useColumnAlignment = True
        
        self.tableView = tableView
        
        tableView.setSelectionBehavior( QTableView.SelectRows );
        
        self.initModel()
    pass

    @abstractmethod
    def initModel( self ):
        self.tableHeader = None
        self.psQuery = None 
    pass

    # adjustColumnWidth
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
    pass
    # -- adjustColumnWidth

    # repaintTableView
    def repaintTableView(self):
        # set wait cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        tableView = self.tableView 
        
        tableView.clearSelection()
        
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()
        
        tableView.repaint()
        tableView.update()
        
        # restore cursor
        QApplication.restoreOverrideCursor()
    pass
    # -- repaintTableView
    
    # searchBtnClicked
    def searchBtnClicked( self , window ):
        self.doSearch( window, 1 )
    pass
    
    def doSearch(self, window, pageNo ):
        
        if True :
            # clear previous search result
            window.status.setText( "검색중입니다. 잠시만 기다려 주세요." )
            
            self.psQuery = None 
            self.repaintTableView() 
        pass
        
        self.psQuery = self.getPsQuerySearchImpl( window, pageNo )
        
        if not self.psQuery : 
            window.status.setText( "" )
        elif self.psQuery : 
            rowCount = self.psQuery.rowCount  
            
            if 1 > rowCount :
                window.status.setText( "검색 결과가 없습니다." )
                
                if hasattr(window, "select" ) : 
                    window.select.setEnabled( False )
                pass
            else :
                window.status.setText( "총 검색 건수: %s" % ( "{:,d}".format( rowCount ) ) )
                
                if hasattr(window, "select" ) : 
                    window.select.setEnabled( True )
                pass
            pass
        pass
    
        self.repaintTableView() 
    pass
    # -- searchBtnClicked
    
    @abstractmethod
    def getPsQuerySearchImpl(self, window, pageNo ):
        pass
    pass
    
    # initSearchBtnClicked
    def initSearchBtnClicked(self , window ):
        self.initModel()
        
        self.repaintTableView() 
    pass
    # -- initSearchBtnClicked

    # getTableHeader
    def getTableHeader(self):
        if hasattr(self, "table") :
            return self.table.getTableHeader()
        else : 
            return self.erpTable.getTableHeader()
        pass
    pass
    # -- getTableHeader
    
    # rowCount
    def rowCount(self, parent):
        debug = False 
        
        psQuery = self.psQuery
        
        count = 0 
        
        if psQuery is None :
            count =  0
        else :
            itemList = psQuery.itemList
            
            if itemList is None :
                count = 0
            else :         
                count = len( itemList )
            pass
        pass
    
        debug and log.use and log.info( "row count = %s" % count )
        
        return count 
    pass
    # -- rowCount

    # columnCount
    def columnCount(self, parent):
        return 1 + len( self.getTableHeader() ) + 1
    pass
    # -- columnCount
    
    # headerData
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 == col :
                return "No."
            elif len( self.getTableHeader() ) >= col :
                col = self.getTableHeader()[ col - 1 ]
                
                if isinstance( col, PsCol ) :
                    return col.dispColName  
                else :
                    return col
                pass
            else :
                return ""
            pass
        else :
            return None
        pass
    pass
    # -- headerData
    
    # flags
    def flags(self, index):
        f = None 
        if index.isValid():
            row = index.row()
            col = index.column()
            
            if 0 < row and 0 == col :
                f = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable 
            else :
                f = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
            pass
        pass 
    
        f = super().flags(index) | f 
    
        return f 
    pass
    # -- flags
    
    # getBackgroundBrush
    def getBackgroundBrush(self , index ):
        row = index.row()
        
        if row%2 :
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
    
    # getItemAt
    def getItemAt(self, row) :
        
        if self.psQuery and self.psQuery.itemList :
            return self.psQuery.itemList[ row ]
        else :
            return None
        pass
    pass
    # -- getItemAt

    # getItemColumnValue
    def getItemColumnValue(self, item , col ):
        '''
        debug = True     
        funName = self.getFunName( inspect.currentframe() )     
        debug and log.use and log.info( funName )
        '''
        
        return item.getColumnValue( col - 1 ) 
    pass
    # -- getItemColumnValue
    
    # data
    def data(self, index, role):        
        if not index.isValid():
            return None
        pass
    
        if role == Qt.BackgroundRole :
            return self.getBackgroundBrush( index )
        elif role == Qt.ForegroundRole :
            return self.getForegroundBrush( index )
        elif role in ( Qt.EditRole, Qt.DisplayRole , Qt.CheckStateRole , Qt.TextAlignmentRole ) :
            row = index.row()
            col = index.column()
                
            item = self.getItemAt( row )
            
            if item is None :
                value = None
            else :
                if Qt.CheckStateRole == role and 0 == col:
                    useSelector = self.useSelector
                    if useSelector :
                        value = ( Qt.Unchecked, Qt.Checked ) [ item.isChecked ]
                    else : 
                        value = None
                    pass 
                elif Qt.CheckStateRole == role :
                    value = None
                elif 0 == col :
                    psQuery = self.psQuery
                    value = ( psQuery.pageNo - 1 )*psQuery.pageRowCount + row + 1
                else :
                    value = self.getItemColumnValue(item, col)
                    
                    valueType = type( value )
                    
                    # convert to its literal value
                    if value is None :
                        value = ""
                    elif valueType not in [ int , float, str ] :
                        value = "%s" % value 
                    pass
                pass
            pass
        
            # return cell alignment value
            if Qt.TextAlignmentRole == role :
                if type( value ) == int :
                    return  Qt.AlignRight | QtCore.Qt.AlignVCenter
                elif type( value ) == float :
                    return  Qt.AlignRight | QtCore.Qt.AlignVCenter
                elif isinstance( value, datetime.date) :
                    return Qt.AlignHCenter | QtCore.Qt.AlignVCenter
                else :
                    if self.useColumnAlignment :
                        return self.getColumnAlignment(col)
                    else :
                        return Qt.AlignLeft | QtCore.Qt.AlignVCenter
                    pass
                pass
            pass
            # -- return cell alignment value
        
            return value
        else :
            return None
        pass
    pass
    # -- data
    
    # getColumnAlignment
    def getColumnAlignment(self, col): 
        debug = False     
        funName = self.getFunName( inspect.currentframe() )     
        debug and log.use and log.info( funName )
        
        alignment = Qt.AlignLeft | QtCore.Qt.AlignVCenter
            
        psQuery = self.psQuery
        
        if psQuery is None or psQuery.itemList is None :
            alignment = Qt.AlignLeft | QtCore.Qt.AlignVCenter
        else :
            itemList = psQuery.itemList
            
            if not hasattr( psQuery, "columnAlignments" ) :
                psQuery.columnAlignments = { }
            pass
        
            columnAlignments = psQuery.columnAlignments 
            
            if col in columnAlignments :
                alignment = columnAlignments[ col ]
            else :
                dataLen = None
                
                alignment = Qt.AlignHCenter | QtCore.Qt.AlignVCenter
                
                diffCount = 0 
                
                itemListLength = len( itemList )
                 
                for item in itemList :
                    data = self.getItemColumnValue(item, col)
                    
                    if not data :
                        currDataLen = 0 
                    else : 
                        data = "%s" % data 
                        data = data.strip()
                        currDataLen = len( data )
                    pass
                
                    debug and log.use and log.info( "data=%s, dataLen=%s" % (data, currDataLen ) )
                
                    if 0 == currDataLen :
                        pass 
                    elif dataLen is None :
                        dataLen = currDataLen
                    elif dataLen != currDataLen :
                        diff = abs( dataLen - currDataLen )
                        
                        if 3 < diff :
                            diffCount += 1
                        else :
                            dataLen = max( dataLen, currDataLen )
                        pass
                    
                        if 0.1 < diffCount/itemListLength :
                            alignment = Qt.AlignLeft | QtCore.Qt.AlignVCenter
                            break 
                        pass
                    
                    pass
                pass
                
                columnAlignments[ col ] = alignment
            pass
            
        pass
    
        debug and log.use and log.info( "Done. " + funName )
        
        return alignment
    pass        
    # -- getColumnAlignment
    
    # -- setData
    def setData(self, index, value, role):
        if not index.isValid():
            return False
        pass
    
        if role == QtCore.Qt.CheckStateRole and index.column() == 0:
            pass
        else:
            pass
        pass 
    
        return False
    pass
    # -- setData

pass