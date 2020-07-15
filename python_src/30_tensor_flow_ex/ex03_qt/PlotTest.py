# -*- coding: utf-8 -*-

import os, sys, numpy as np , matplotlib

from PyQt5 import QtCore, QtWidgets, uic 

matplotlib.use('QT5Agg')

import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        uic.loadUi( './PlotTest.ui', self ) 
        
        # test data
        data = np.array([0.7,0.7,0.7,0.8,0.9,0.9,1.5,1.5,1.5,1.5])        
        fig, ax1 = plt.subplots()
        bins = np.arange(0.6, 1.62, 0.02)
        n1, bins1, patches1 = ax1.hist(data, bins, alpha=0.6, density=False, cumulative=False)
        
        # plot
        self.figureCanvas = FigureCanvas( fig )
        lay = QtWidgets.QVBoxLayout(self.content_plot)  
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(self.figureCanvas)
        
        # add toolbar
        use_toolbar = 1
        use_toolbar and self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.figureCanvas, self))
    pass
pass

if __name__ == '__main__':
    print( "Pwd 1: %s" % os.getcwd())
    # change working dir to current file
    dirname = os.path.dirname(__file__)
    print( "dirname: ", dirname )
    dirname and os.chdir(dirname)
    print( "Pwd 2: %s" % os.getcwd())

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
pass