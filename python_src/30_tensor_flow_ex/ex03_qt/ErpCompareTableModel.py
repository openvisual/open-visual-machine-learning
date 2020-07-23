# -*- coding: utf-8 -*-

from psCom.PsCom import log
from psModel.PsTableModel import PsTableModel

# ErpCompareTableModel
class ErpCompareTableModel( PsTableModel ): 
    
    debug = True 
    
    def __init__(self, parent, tableView , erpTable ):
        super().__init__( parent, tableView )
        
        self.erpTable = erpTable 
    pass

    def initModel( self ):
        self.psQuery = None 
    pass

    def getPsQuerySearchImpl(self, window, pageNo ):
        debug = self.debug
        
        erpTable = self.erpTable
        
        e_property_no   = window.e_property_no.text().strip()
        e_lot_no        = window.e_lot_no.text().strip()
        e_gu_name       = window.e_gu_name.valueList[ window.e_gu_name.currentIndex() ].value
        
        erpTable.e_property_no  = e_property_no
        erpTable.e_lot_no       = e_lot_no
        erpTable.e_gu_name      = e_gu_name
        
        debug and log.use and log.info( "e_gu_name = %s" % e_gu_name )
        
        psQuery = erpTable.getErpComparePsQuery( pageNo = pageNo, pageRowCount = 100_000, debug = debug )
        
        return psQuery
    pass

pass

# // 
    