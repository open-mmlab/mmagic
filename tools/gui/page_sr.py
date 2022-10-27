# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
from component import ConcatImageWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from utils import layout2widget


class PatchTab(QtWidgets.QWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.statusBar = self.parent.statusBar
        self.file_paths = []
        self.labels = []
        self.images = None
        self.isShow = False

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
        self.tb_files.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.tb_files.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.tb_files.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tb_files.customContextMenuRequested.connect(self.tableMenu)

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
        # select mode
        self.modeRect = QtWidgets.QGroupBox('Select Mode')
        self.modeRect.setFlat(True)
        # self.modeRect.setStyleSheet(styleSheet)
        btn_mode1 = QtWidgets.QRadioButton('Input whole image', self.modeRect)
        btn_mode2 = QtWidgets.QRadioButton('Input crop image', self.modeRect)
        self.btnGroup_mode = QtWidgets.QButtonGroup()
        self.btnGroup_mode.addButton(btn_mode1, 0)
        self.btnGroup_mode.addButton(btn_mode2, 1)
        self.btnGroup_mode.button(0).setChecked(True)
        self.btnGroup_mode.idToggled.connect(self.reset)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(btn_mode1)
        hlayout.addWidget(btn_mode2)
        self.modeRect.setLayout(hlayout)

        # select dirType
        self.dirTypeRect = QtWidgets.QGroupBox('Select File or Directory')
        self.dirTypeRect.setFlat(True)
        # self.dirTypeRect.setStyleSheet(styleSheet)
        btn_dirType1 = QtWidgets.QRadioButton('Directory', self.dirTypeRect)
        btn_dirType2 = QtWidgets.QRadioButton('File', self.dirTypeRect)
        self.btnGroup_dirType = QtWidgets.QButtonGroup()
        self.btnGroup_dirType.addButton(btn_dirType1, 0)
        self.btnGroup_dirType.addButton(btn_dirType2, 1)
        self.btnGroup_dirType.button(0).setChecked(True)
        self.btnGroup_dirType.idToggled.connect(self.reset)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(btn_dirType1)
        hlayout.addWidget(btn_dirType2)
        self.dirTypeRect.setLayout(hlayout)

        # select gt
        self.cb_gt = QtWidgets.QComboBox()
        self.btn_setGt = QtWidgets.QPushButton()
        self.btn_setGt.setText('Select GT')
        self.btn_setGt.clicked.connect(self.open_gt)

        # set column
        self.spin_cols = QtWidgets.QSpinBox()
        self.spin_cols.setMinimum(1)
        self.spin_cols.setMaximum(10)
        self.spin_cols.setValue(4)

        # set scale
        self.txt_scale = QtWidgets.QLabel('100 %')
        self.slider_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_scale.setMinimum(0)
        self.slider_scale.setMaximum(200)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.set_scale)

        # operation
        self.btn_run = QtWidgets.QPushButton()
        self.btn_run.setText('Run')
        self.btn_run.clicked.connect(self.run)
        self.btn_reset = QtWidgets.QPushButton()
        self.btn_reset.setText('Reset')
        self.btn_reset.clicked.connect(self.reset)
        self.btn_save = QtWidgets.QPushButton()
        self.btn_save.setText('Save')
        self.btn_save.clicked.connect(self.save)

        right_grid = QtWidgets.QGridLayout()
        right_grid.addWidget(self.modeRect, 0, 0, 1, 20)
        right_grid.addWidget(self.dirTypeRect, 1, 0, 1, 20)
        right_grid.addWidget(self.cb_gt, 2, 0, 1, 15)
        right_grid.addWidget(self.btn_setGt, 2, 15, 1, 5)
        right_grid.addWidget(self.spin_cols, 3, 0, 1, 15)
        right_grid.addWidget(QtWidgets.QLabel('Set   columns'), 3, 16, 1, 3)
        right_grid.addWidget(self.txt_scale, 4, 0, 1, 1)
        right_grid.addWidget(self.slider_scale, 4, 1, 1, 14)
        right_grid.addWidget(QtWidgets.QLabel('Set   scale'), 4, 16, 1, 3)
        right_grid.addWidget(self.btn_run, 5, 0, 1, 20)
        right_grid.addWidget(self.btn_reset, 6, 0, 1, 20)
        right_grid.addWidget(self.btn_save, 7, 0, 1, 20)

        # Bottom Widget
        self.image_scroll = QtWidgets.QScrollArea()

        # Splitter
        hsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        vsplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        hsplitter.addWidget(layout2widget(left_grid))
        hsplitter.addWidget(layout2widget(right_grid))
        hsplitter.setStretchFactor(0, 1)
        hsplitter.setStretchFactor(1, 2)
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(self.image_scroll)
        vsplitter.setStretchFactor(0, 1)
        vsplitter.setStretchFactor(1, 5)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(vsplitter)
        self.setLayout(hlayout)

    def open_file(self):
        """Open a file or directory from dialog."""
        if self.btnGroup_dirType.checkedId() == 0:
            path = QtWidgets.QFileDialog.getExistingDirectory()
            label = path.split('/')[-1]
            self.input_file.setText(path)
            self.input_label.setText(label)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Select file', '', 'Images (*.jpg *.png)')
            label = path.split('/')[-1].split('.')[0]
            self.input_file.setText(path)
            self.input_label.setText(label)

    def add_file(self):
        """Add opened file or directory to table."""
        if os.path.exists(self.input_file.text()):
            row = self.tb_files.rowCount()
            self.tb_files.setRowCount(row + 1)
            self.tb_files.setItem(
                row, 0, QtWidgets.QTableWidgetItem(self.input_file.text()))
            self.tb_files.setItem(
                row, 1, QtWidgets.QTableWidgetItem(self.input_label.text()))
            if self.cb_gt.count() == 0:
                self.set_gt(self.input_file.text())
        else:
            QtWidgets.QMessageBox.about(self, 'Message',
                                        'Please input available file/folder')

    def tableMenu(self, pos):
        """Set mouse right button menu of table."""
        menu = QtWidgets.QMenu()
        menu_up = menu.addAction('move up')
        menu_down = menu.addAction('move down')
        menu_delete = menu.addAction('delete')
        menu_gt = menu.addAction('set gt')
        menu_pos = self.tb_files.mapToGlobal(pos)
        for i in self.tb_files.selectionModel().selection().indexes():
            row = i.row()
        action = menu.exec(menu_pos)
        if action == menu_up:
            self.tb_swap(row, -1)
        elif action == menu_down:
            self.tb_swap(row, 1)
        elif action == menu_delete:
            self.tb_files.removeRow(row)
        elif action == menu_gt:
            self.set_gt(self.tb_files.item(row, 0).text())
        if self.isShow:
            self.run()

    def tb_swap(self, row, move):
        """Move items of table."""
        if move == -1:
            if 0 == row:
                return
            target = row - 1
        elif move == 1:
            if self.tb_files.rowCount() - 1 == row:
                return
            target = row + 1
        for col in range(self.tb_files.columnCount()):
            tmp = self.tb_files.takeItem(row, col)
            self.tb_files.setItem(row, col,
                                  self.tb_files.takeItem(target, col))
            self.tb_files.setItem(target, col, tmp)

    def open_gt(self):
        """Open GT file from dialog."""
        if self.btnGroup_dirType.checkedId() == 0:
            path = QtWidgets.QFileDialog.getExistingDirectory()
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Select gt file', '', 'Images (*.jpg *.png)')
        self.set_gt(path)

    def set_gt(self, path):
        """Set GT ComboBox items."""
        self.cb_gt.clear()
        if os.path.isfile(path):
            self.cb_gt.addItem(path)
        else:
            files = sorted(os.listdir(path))
            for f in files:
                self.cb_gt.addItem(path + '/' + f)

    def set_scale(self):
        """Set scale."""
        scale = self.slider_scale.value()
        self.txt_scale.setText(f'{scale} %')
        if self.isShow:
            self.run()

    def run(self):
        """Generate patch compare result."""
        if self.cb_gt.currentText() == '':
            QtWidgets.QMessageBox.about(self, 'Message', 'Please set gt!')
            return

        rows = self.tb_files.rowCount()
        if rows <= 0:
            QtWidgets.QMessageBox.about(self, 'Message',
                                        'Please add a file at least!')
            return

        files = []
        self.labels = []
        for r in range(rows):
            files.append(self.tb_files.item(r, 0).text())
            self.labels.append(self.tb_files.item(r, 1).text())

        file_name = os.path.basename(self.cb_gt.currentText())
        self.file_paths = []
        for i, file in enumerate(files):
            if self.btnGroup_dirType.checkedId() == 0:
                self.file_paths.append(file + '/' + file_name)
            else:
                self.file_paths.append(file)

        mode = self.btnGroup_mode.checkedId()
        self.images = ConcatImageWidget(self, mode, self.spin_cols.value())
        self.images.set_images(self.file_paths, self.labels,
                               self.cb_gt.currentText(),
                               self.slider_scale.value() / 100.0)
        self.image_scroll.setWidget(self.images)
        self.isShow = True

    def reset(self):
        """Init window."""
        self.file_paths = []
        self.labels = []
        self.isShow = False

        self.input_label.clear()
        self.input_file.clear()
        self.tb_files.setRowCount(0)
        self.cb_gt.clear()
        self.spin_cols.setValue(4)
        self.slider_scale.setValue(100)
        self.images = ConcatImageWidget(self, 0, self.spin_cols.value())
        self.image_scroll.setWidget(self.images)

    def save(self):
        """Save patch compare result."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'save')
        if self.images:
            self.images.grab().save(path)
            QtWidgets.QMessageBox.about(self, 'Message', 'Success!')
        else:
            QtWidgets.QMessageBox.about(self, 'Message', 'Nothing to save.')


class SliderTab(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.images = [None, None]

        self.dirTypeRect = QtWidgets.QGroupBox('Select File or Directory')
        self.dirTypeRect.setFlat(True)
        btn_dirType1 = QtWidgets.QRadioButton('Directory', self.dirTypeRect)
        btn_dirType2 = QtWidgets.QRadioButton('File', self.dirTypeRect)
        self.btnGroup_dirType = QtWidgets.QButtonGroup()
        self.btnGroup_dirType.addButton(btn_dirType1, 0)
        self.btnGroup_dirType.addButton(btn_dirType2, 1)
        self.btnGroup_dirType.button(0).setChecked(True)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(btn_dirType1)
        hlayout.addWidget(btn_dirType2)
        self.dirTypeRect.setLayout(hlayout)

        self.cb_1 = QtWidgets.QComboBox()
        self.cb_2 = QtWidgets.QComboBox()
        self.cb_1.currentIndexChanged.connect(self.change_image)
        self.cb_2.currentIndexChanged.connect(self.change_image)
        self.input_label_1 = QtWidgets.QLineEdit()
        self.input_label_2 = QtWidgets.QLineEdit()
        self.input_title = QtWidgets.QLineEdit()

        self.btn_add_1 = QtWidgets.QPushButton()
        self.btn_add_2 = QtWidgets.QPushButton()
        self.btn_add_1.setText('Add a file')
        self.btn_add_2.setText('Add a directory')
        self.btn_add_1.clicked.connect(self.add_1)
        self.btn_add_2.clicked.connect(self.add_1)

        left_grid = QtWidgets.QGridLayout()
        left_grid.addWidget(self.dirTypeRect, 0, 0, 1, 10)
        left_grid.addWidget(QtWidgets.QLabel('Set image 1'), 1, 0, 1, 1)
        left_grid.addWidget(self.cb_1, 1, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Set image 2'), 2, 0, 1, 1)
        left_grid.addWidget(self.cb_2, 2, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Set label 1'), 3, 0, 1, 1)
        left_grid.addWidget(self.input_label_1, 3, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Set label 2'), 4, 0, 1, 1)
        left_grid.addWidget(self.input_label_2, 4, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Set title'), 5, 0, 1, 1)
        left_grid.addWidget(self.input_title, 5, 1, 1, 9)
        left_grid.addWidget(self.btn_add_1, 6, 0, 1, 5)
        left_grid.addWidget(self.btn_add_2, 6, 5, 1, 5)

        self.image_scroll = QtWidgets.QScrollArea()
        self.image_scroll.setAlignment(QtCore.Qt.AlignCenter)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.show_image)
        right_grid = QtWidgets.QGridLayout()
        right_grid.addWidget(self.image_scroll, 0, 0)
        right_grid.addWidget(self.slider, 1, 0)

        # Splitter
        hsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        hsplitter.addWidget(layout2widget(left_grid))
        hsplitter.addWidget(layout2widget(right_grid))
        hsplitter.setStretchFactor(0, 1)
        hsplitter.setStretchFactor(1, 5)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(hsplitter)
        self.setLayout(hlayout)

    def add_1(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select gt file', '', 'Images (*.jpg *.png *.mp4 *.avi)')
        if self.cb_1.count() > 0:
            if self.cb_1.findText(path) > -1:
                self.cb_1.removeItem(self.cb_1.findText(path))
            if self.cb_2.findText(path) > -1:
                self.cb_2.removeItem(self.cb_2.findText(path))
            self.cb_1.addItem(path)
            self.cb_2.addItem(path)
            self.cb_2.setCurrentIndex(self.cb_2.count() - 1)
        else:
            self.cb_1.addItem(path)
            self.cb_2.addItem(path)
            self.cb_1.setCurrentIndex(0)
            self.cb_2.setCurrentIndex(0)

    def change_image(self):
        self.images[0] = cv2.imread(self.cb_1.currentText())
        self.images[1] = cv2.imread(self.cb_2.currentText())
        self.show_image()

    def show_image(self):
        if self.images[0] is None or self.images[1] is None:
            return
        img1, img2 = self.images
        h2, w2, c2 = img2.shape
        self.slider.setMaximum(w2)
        # img2 = cv2.resize(img2, (800, int(800 / w2 * h2)))
        h2, w2, c2 = img2.shape
        img1 = cv2.resize(img1, (w2, h2))
        v = self.slider.value()
        img11 = img1[:, 0:v].copy()
        img22 = img2[:, v:].copy()
        img = np.hstack((img11, img22))
        img = cv2.line(img, (v, 0), (v, h2), (0, 222, 0), 4)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_dis = QtGui.QImage(rgb_img, w2, h2, w2 * c2,
                               QtGui.QImage.Format_RGB888)
        jpg = QtGui.QPixmap.fromImage(img_dis)
        # .scaled(
        # self.imageArea.width(), int(self.imageArea.width() / w2 * h2))

        imageArea = QtWidgets.QLabel()
        imageArea.setFrameShape(QtWidgets.QFrame.Box)
        imageArea.setLineWidth(2)
        imageArea.setAlignment(QtCore.Qt.AlignBottom)
        imageArea.setStyleSheet(
            'border-width: 0px; border-style: solid; border-color: rgb(100, 100, 100);background-color: rgb(255, 255, 255)'  # noqa
        )
        imageArea.setPixmap(jpg)
        self.image_scroll.setWidget(imageArea)


class SRPage(QtWidgets.QWidget):

    def __init__(self, parent) -> None:
        super().__init__()

        self.parent = parent
        self.statusBar = self.parent.statusBar

        self.tab_patch = PatchTab(self)
        self.tab_slider = SliderTab()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_patch, 'patch compare')
        self.tabs.addTab(self.tab_slider, 'before/after slider')

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.tabs)
        # self.tabs.currentChanged.connect(self.tabsCurrentChanged)
