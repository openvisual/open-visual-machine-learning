# -*- coding: utf-8 -*-

import os, sys
from PyQt5 import QtWidgets, uic

class MyQtApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__() # Call the inherited classes __init__ method
        
        uic.loadUi( './MyQtApp.ui', self) # Load the .ui file

        # signal -> slot 연결
        self.myPushButton.clicked.connect( self.when_MyPushButton_clicked )

        self.myDial.valueChanged.connect( self.myLcdNumber.display)

        self.show() # Show the GUI
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
    print( "dirname: ", dirname )
    dirname and os.chdir(dirname)
    print( "Pwd 2: %s" % os.getcwd())

    # Create an instance of QtWidgets.QApplication
    app = QtWidgets.QApplication(sys.argv)
    window = MyQtApp() # Create an instance of our class
    app.exec_() # Start the application
pass

