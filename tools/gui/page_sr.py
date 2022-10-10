# Copyright (c) OpenMMLab. All rights reserved.
import os

from PyQt5 import QtCore, QtWidgets

from tools.gui.component import ConcatImageWidget
from tools.gui.utils import layout2widget


class PatchTab(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.file_paths = []
        self.labels = []
        self.dirs = ''
        self.images = None

        # Left Widget
        self.btn_add_file = QtWidgets.QPushButton()
        self.btn_add_file.setText('Add')
        self.btn_add_file.clicked.connect(self.add_file)
        self.btn_open_file = QtWidgets.QPushButton()
        self.btn_open_file.setText('Open new')
        self.btn_open_file.clicked.connect(self.open_file)

        self.input_label = QtWidgets.QLineEdit()
        self.input_file = QtWidgets.QLineEdit()

        self.tb_files = QtWidgets.QTableWidget()
        self.tb_files.setColumnCount(3)
        self.tb_files.setHorizontalHeaderLabels(
            ['File/Folder', 'Label', 'Other'])
        self.tb_files.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)

        left_grid = QtWidgets.QGridLayout()
        left_grid.addWidget(self.tb_files, 0, 0, 5, 10)
        left_grid.addWidget(self.input_file, 6, 0, 1, 4)
        left_grid.addWidget(self.input_label, 6, 4, 1, 2)
        left_grid.addWidget(self.btn_open_file, 6, 6, 1, 2)
        left_grid.addWidget(self.btn_add_file, 6, 8, 1, 2)

        # Right Widget
        styleSheet = '''
            QGroupBox {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #E0E0E0, stop: 1 #FFFFFF);
                border: 1px solid #999999;
                border-radius: 5px;
                margin-top: 2ex;  /*leave space at the top for the title */
                padding: 2ex 10ex;
                font-size: 20px;
                color: black;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;    /* position at the top center */
                padding: 0 3px;
                left: 30px;
                font-size: 8px;
                color: black;
            }
        ''' # noqa
        self.modeRect = QtWidgets.QGroupBox('Mode')
        self.modeRect.setFlat(True)
        self.modeRect.setStyleSheet(styleSheet)
        btn_mode1 = QtWidgets.QRadioButton('input crop image', self.modeRect)
        btn_mode2 = QtWidgets.QRadioButton('input whole image', self.modeRect)
        self.btnGroup_mode = QtWidgets.QButtonGroup()
        self.btnGroup_mode.addButton(btn_mode1)
        self.btnGroup_mode.addButton(btn_mode2)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(btn_mode1)
        hlayout.addWidget(btn_mode2)
        self.modeRect.setLayout(hlayout)

        self.dirTypeRect = QtWidgets.QGroupBox('File or Directory')
        self.dirTypeRect.setFlat(True)
        self.dirTypeRect.setStyleSheet(styleSheet)
        btn_dirType1 = QtWidgets.QRadioButton('File', self.dirTypeRect)
        btn_dirType2 = QtWidgets.QRadioButton('Directory', self.dirTypeRect)
        self.btnGroup_dirType = QtWidgets.QButtonGroup()
        self.btnGroup_dirType.addButton(btn_dirType1)
        self.btnGroup_dirType.addButton(btn_dirType2)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(btn_dirType1)
        hlayout.addWidget(btn_dirType2)
        self.dirTypeRect.setLayout(hlayout)

        self.input_gt = QtWidgets.QLineEdit()
        self.btn_setGt = QtWidgets.QPushButton()
        self.btn_setGt.setText('Select GT')
        self.btn_setGt.clicked.connect(self.setGt)

        self.btn_run = QtWidgets.QPushButton()
        self.btn_run.setText('Run')
        self.btn_run.clicked.connect(self.run)
        self.btn_save = QtWidgets.QPushButton()
        self.btn_save.setText('Save')
        self.btn_save.clicked.connect(self.save)
        right_grid = QtWidgets.QGridLayout()
        right_grid.addWidget(self.modeRect, 0, 0, 1, 20)
        right_grid.addWidget(self.dirTypeRect, 1, 0, 1, 20)
        right_grid.addWidget(self.input_gt, 2, 0, 1, 15)
        right_grid.addWidget(self.btn_setGt, 2, 15, 1, 5)
        right_grid.addWidget(self.btn_run, 3, 0, 1, 20)
        right_grid.addWidget(self.btn_save, 4, 0, 1, 20)

        # Bottom Widget
        self.image_scroll = QtWidgets.QScrollArea()

        # Splitter
        hsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        vsplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        hsplitter.addWidget(layout2widget(left_grid))
        hsplitter.addWidget(layout2widget(right_grid))
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(self.image_scroll)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(vsplitter)
        self.setLayout(hlayout)

    def open_file(self):
        path = QtWidgets.QFileDialog.getExistingDirectory()
        label = path.split('/')[-1]
        self.input_file.setText(path)
        self.input_label.setText(label)

    def add_file(self):
        row = self.tb_files.rowCount()
        self.tb_files.setRowCount(row + 1)
        self.tb_files.setItem(
            row, 0, QtWidgets.QTableWidgetItem(self.input_file.text()))
        self.tb_files.setItem(
            row, 1, QtWidgets.QTableWidgetItem(self.input_label.text()))
        # path = QFileDialog.getExistingDirectory()
        # self.dirs += path + '\n'
        # self.folder_list.setPlainText(self.dirs)

    def run(self):
        if self.input_gt.text() == '':
            QtWidgets.QMessageBox.about(self, 'Message', 'Please set gt!')
            return
        files = []
        self.labels = []
        rows = self.tb_files.rowCount()
        if rows <= 0:
            QtWidgets.QMessageBox.about(self, 'Message',
                                        'Please add a file at least!')
            return
        for r in range(rows):
            files.append(self.tb_files.item(r, 0).text())
            self.labels.append(self.tb_files.item(r, 1).text())
        file_names = sorted(os.listdir(files[0]))

        self.file_paths = []
        for i, file in enumerate(files):
            self.file_paths.append(file + '/' + file_names[0])
        self.images = ConcatImageWidget()
        self.images.set_images(self.file_paths, self.labels,
                               self.input_gt.text())
        self.image_scroll.setWidget(self.images)

    def save(self):
        if self.images:
            self.images.grab().save('1.png')
            QtWidgets.QMessageBox.about(self, 'Message', 'Success!')
        else:
            QtWidgets.QMessageBox.about(self, 'Message', 'Nothing to save.')

    def setGt(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select gt file', '', 'Images (*.jpg *.png)')
        self.input_gt.setText(path)


class SRPage(QtWidgets.QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.tab_patch = PatchTab()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_patch, 'patch')

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.tabs)
        # self.tabs.currentChanged.connect(self.tabsCurrentChanged)
