# -*- coding: utf-8 -*-

import os 
import inspect

from qgis.PyQt import uic

from qgis.PyQt.QtCore import QEvent 
from qgis.PyQt.QtWidgets import QDialog 

from psCom.PsCom import PsCom, log 

from PyQt5.QtCore import QPoint
from psModel.ErpCompareTableModel import ErpCompareTableModel
from psModel.ErpRecordTableModel import ErpRecordTableModel
from psProject.PsProjectKd import PsProjectKd

from psModel.psErp.PsErpValve import PsErpValve
from psModel.psErp.PsErpTestBox import PsErpTestBox
from psModel.psErp.PsErpLocateBox import PsErpLocateBox
from psModel.psErp.PsErpInspect import PsErpInspect
from psModel.psErp.PsErpRegulator import PsErpRegulator
from psModel.psErp.PsErpRectifier import PsErpRectifier

FORM_CLASS, _ = uic.loadUiType(os.path.join( os.path.dirname(__file__), "Ui_ErpCompare.ui" ) )

class Ui_ErpCompareImpl( QDialog, FORM_CLASS , PsCom ):
    
    debug = True 
    
    # init
    def __init__(self, parent ):
        super(QDialog, self).__init__(parent)
        
        self.parent = parent
        
        self.def_property_no = ""
        
        self.setupUi(self)
        
        if True :
            # set main table view        
            tableView = self.tableView
            
            erpTable = PsErpValve()  
            
            parent = self 
            
            tableModel = ErpCompareTableModel( parent, tableView , erpTable ) 
            
            tableView.setModel( tableModel )
            
            tableModel.adjustColumnWidth()
            
            tableView.doubleClicked.connect( self.selectBtnClicked )
            
            tableView.clicked.connect( self.tableViewClicked )
            
            self.tableModel = tableModel
            
            # -- set main table view 
        pass
    
        if True :
            # set erp record info record view
            tableView_erpRecord = self.tableView_erpRecord
            
            parent = self 
            erpRecordTableModel = ErpRecordTableModel( parent, tableView_erpRecord )
            
            tableView_erpRecord.setModel( erpRecordTableModel )
            
            erpRecordTableModel.adjustColumnWidth()
            
            self.erpRecordTableModel = erpRecordTableModel
            
            # -- set erp record info record view 
        pass
    
        if True :
            psProject = PsProjectKd() 
            valueList = psProject.getHangJungGuList( includeEmptyItem = True)
            
            widget = self.e_gu_name
            
            widget.valueList = valueList
            
            for idx, item in enumerate( valueList ) :
                widget.insertItem( idx, item.value ) 
            pass
        pass
        
        # radio button slot connect
        self.valve.clicked.connect( lambda: self.changeErpTableBtnClicked( PsErpValve() ) )
        self.testBox.clicked.connect( lambda: self.changeErpTableBtnClicked( PsErpTestBox() ) )
        self.locateBox.clicked.connect( lambda: self.changeErpTableBtnClicked( PsErpLocateBox() ) )
        self.inspect.clicked.connect( lambda: self.changeErpTableBtnClicked( PsErpInspect() ) )
        self.regulator.clicked.connect( lambda: self.changeErpTableBtnClicked( PsErpRegulator() ) )
        self.rectifier.clicked.connect( lambda: self.changeErpTableBtnClicked( PsErpRectifier() ) )
        # -- radio button slot connect
        
        self.search.clicked.connect( self.searchBtnClicked )
        self.initSearch.clicked.connect( self.initSearchBtnClicked )
        
        self.select.clicked.connect( self.selectBtnClicked )
        
        self.installEventFilter( self )
        
    # -- init
    
    # showEvent
    def show(self ):
        super( Ui_ErpCompareImpl, self ).show()
        
        widget = self.parent 
        
        # calculate the botoom right point from the parents rectangle
        topRight = widget.rect().topLeft()

        # map that point as a global position
        global_point = widget.mapToGlobal( topRight )

        self.move( global_point + QPoint( 5, 0 ) )
    pass
    # -- showEvent
    
    # tableViewClicked
    def tableViewClicked(self):
        debug = True     
        funName = self.getFunName( inspect.currentframe() )   
        debug and log.use and log.info( funName )
        
        tableView = self.tableView
        
        indexes = tableView.selectedIndexes()
        
        if 1 > len( indexes ) : 
            self.select.setEnabled( False )
        else :
            self.select.setEnabled( True )
            
            row = indexes[0].row()
            
            tableModel = self.tableModel
            
            if tableModel.psQuery and tableModel.psQuery.itemList :
                erpRecord = tableModel.psQuery.itemList[ row ]
                
                debug and log.use and log.info( "sel item = %s" % erpRecord )
                
                erpRecordTableModel = self.erpRecordTableModel 
                
                erpRecordTableModel.initModel( erpRecord ) 
                
                erpRecordTableModel.repaintTableView()
            pass
            
        pass
    pass
    # -- tableViewClicked
    
    # selectBtnClicked
    def selectBtnClicked(self):
        debug = True     
        funName = self.getFunName( inspect.currentframe() )   
        debug and log.use and log.info( funName )
        
        tableView = self.tableView
        
        indexes = tableView.selectedIndexes()
        
        if 0 < len( indexes ) : 
            row = indexes[0].row()
            
            tableModel = self.tableModel
            
            if tableModel.psQuery and tableModel.psQuery.itemList :
                item = tableModel.psQuery.itemList[ row ]
                
                debug and log.use and log.info( "sel item = %s" % item )  
            pass
        pass
    pass
    # -- selectBtnClicked
                
    # searchBtnClicked
    def searchBtnClicked(self):
        # erp record init
        erpRecordTableModel = self.erpRecordTableModel
        erpRecord = None 
        erpRecordTableModel.initModel( erpRecord )
        erpRecordTableModel.repaintTableView()
        # -- erp record init
        
        window = self
        self.tableModel.searchBtnClicked( window )
    pass
    # -- searchBtnClicked
    
    # initSearchBtnClicked
    def initSearchBtnClicked(self):
        tableModel = self.tableModel
        erpTable = tableModel.erpTable 
        
        self.e_property_no.setText( self.def_property_no )
        self.e_lot_no.setText( "" )
        
        if hasattr( erpTable, "e_lot_no" ) :
            self.e_lot_no.setEnabled( True )
        else :
            self.e_lot_no.setEnabled( False )
        pass
        
        self.select.setEnabled( False )
        self.status.setText( "" )
        
        window = self 
        
        tableModel.initSearchBtnClicked( window )
        
        # erp record init
        erpRecordTableModel = self.erpRecordTableModel
        erpRecord = None 
        erpRecordTableModel.initModel( erpRecord )
        erpRecordTableModel.repaintTableView()
        # -- erp record init
    pass
    # -- initSearchBtnClicked
    
    # changeErpTableBtnClicked
    def changeErpTableBtnClicked(self , erpTable ): 
        debug = True     
        funName = self.getFunName( inspect.currentframe() )   
        debug and log.use and log.info( funName )
        
        tableModel = self.tableModel
        
        self.e_property_no.setText( self.def_property_no )
        self.e_lot_no.setText( "" )
        
        if hasattr( erpTable, "e_lot_no" ) :
            self.e_lot_no.setEnabled( True )
        else :
            self.e_lot_no.setEnabled( False )
        pass
        
        self.select.setEnabled( False )
        self.status.setText( "" )
        
        window = self 
        
        tableModel.erpTable = erpTable
        
        debug and log.use and log.info( "erpTable = %s" % tableModel.erpTable )
        
        tableModel.initSearchBtnClicked( window )
        
        # erp record init
        erpRecordTableModel = self.erpRecordTableModel
        erpRecord = None 
        erpRecordTableModel.initModel( erpRecord )
        erpRecordTableModel.repaintTableView()
        # -- erp record init
    pass
    # -- changeErpTableBtnClicked
    
    # eventFilter   
    def eventFilter(self, qobject, qevent) :
        if QEvent.Close == qevent.type() : 
            psMain = self.getPluginMain()
            erpCompareAction = psMain.erpCompareAction
            erpCompareAction.setEnabled( True )
        pass
                
        return super(QDialog, self).eventFilter(qobject, qevent)
    pass
    # -- eventFilter
    
pass 
# //