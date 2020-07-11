# -*- coding: utf-8 -*-

import os
import sys
from PyQt5 import QtWidgets, uic

class MyQtApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi( './myQtApp.ui', self) # Load the .ui file
        self.show() # Show the GUI
    pass
pass

if __name__ == '__main__':
    print( "Pwd 1: %s" % os.getcwd())
    # change working dir to current file
    os.chdir(os.path.dirname(__file__))
    print( "Pwd 2: %s" % os.getcwd())

    # Create an instance of QtWidgets.QApplication
    app = QtWidgets.QApplication(sys.argv)
    window = MyQtApp() # Create an instance of our class
    app.exec_() # Start the application
pass

