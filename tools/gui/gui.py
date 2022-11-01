# Copyright (c) OpenMMLab. All rights reserved.
import sys

from component import QLabelClick
from page_sr import SRPage
from PyQt5 import QtWidgets


class Homepage(QtWidgets.QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        t1 = QLabelClick()
        t2 = QLabelClick()
        t3 = QLabelClick()
        t4 = QLabelClick()
        t1.setText('general')
        t2.setText('sr')
        t3.setText('inpainting')
        t4.setText('matting')
        layout.addWidget(t1, 0, 0)
        layout.addWidget(t2, 0, 1)
        layout.addWidget(t3, 1, 0)
        layout.addWidget(t4, 1, 1)

        t1.clicked.connect(self.main_window.change_window)
        t2.clicked.connect(self.main_window.change_window)
        t3.clicked.connect(self.main_window.change_window)
        t4.clicked.connect(self.main_window.change_window)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('MMEditing Viewer')
        # # MenuBar
        # menubar_Aaa = self.menuBar().addMenu('Aaa')
        # menubar_Bbb = self.menuBar().addMenu('Bbb')
        # menubar_Ccc = self.menuBar().addMenu('Ccc')
        # menubar_Aaa.addAction('New')
        # save = QtWidgets.QAction('Save', self)
        # save.setShortcut('Ctrl+S')
        # menubar_Aaa.addAction(save)
        # menubar_Bbb.addAction('New')
        # menubar_Ccc.addAction('New')

        # # ToolBar
        # self.toolBar = QtWidgets.QToolBar('ToolBar')
        # open = QtWidgets.QAction(QtGui.QIcon(), 'Open', self)
        # save = QtWidgets.QAction(QtGui.QIcon(), 'Save', self)
        # self.toolBar.addAction(open)
        # self.toolBar.addAction(save)
        # self.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, self.toolBar)

        # StatusBar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('')

        self.homepage = Homepage(self)
        self.sr = SRPage(self)
        self.setCentralWidget(self.sr)

    def change_window(self, wname):
        if wname == 'sr':
            self.setCentralWidget(self.sr)
        elif wname == 'general':
            self.setCentralWidget(self.general)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myWin = MainWindow()
    myWin.showMaximized()
    sys.exit(app.exec_())
