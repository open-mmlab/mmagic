# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os

import cv2
from component import ConcatImageWidget, QLabelSlider, VideoSlider
from PyQt5 import QtCore, QtWidgets
from utils import layout2widget


class PatchTab(QtWidgets.QWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.statusBar = self.parent.statusBar
        self.file_paths = []
        self.labels = []
        self.rect = None
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
        self.cb_gt.currentTextChanged.connect(self.change_gt)

        # set column
        self.spin_cols = QtWidgets.QSpinBox()
        self.spin_cols.setMinimum(1)
        self.spin_cols.setMaximum(10)
        self.spin_cols.setValue(4)
        self.spin_cols.valueChanged.connect(self.set_column)

        # set scale
        self.scale = 100
        self.txt_scale = QtWidgets.QLabel('100 %')
        self.slider_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_scale.setMinimum(1)
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
        right_grid.addWidget(QtWidgets.QLabel('Reference'), 2, 16, 1, 3)
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
        self.image_scroll.installEventFilter(self)

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
            if len(path) <= 0:
                return
            label = path.split('/')[-1]
            self.input_file.setText(path)
            self.input_label.setText(label)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Select file', '', 'Images (*.jpg *.png)')
            if len(path) <= 0:
                return
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
            # if self.cb_gt.count() == 0:
            #     self.set_gt(self.input_file.text())
        else:
            QtWidgets.QMessageBox.about(self, 'Message',
                                        'Please input available file/folder')

    def tableMenu(self, pos):
        """Set mouse right button menu of table."""
        menu = QtWidgets.QMenu()
        menu_up = menu.addAction('move up')
        menu_down = menu.addAction('move down')
        menu_delete = menu.addAction('delete')
        menu_gt = menu.addAction('set reference')
        menu_pos = self.tb_files.mapToGlobal(pos)
        row = None
        for i in self.tb_files.selectionModel().selection().indexes():
            row = i.row()
        if row is None or row >= self.tb_files.rowCount():
            return
        action = menu.exec(menu_pos)
        if action == menu_up:
            self.tb_swap(row, -1)
        elif action == menu_down:
            self.tb_swap(row, 1)
        elif action == menu_delete:
            self.tb_files.removeRow(row)
        elif action == menu_gt:
            for r in range(self.tb_files.rowCount()):
                self.tb_files.setItem(r, 2, QtWidgets.QTableWidgetItem(''))
            self.tb_files.setItem(row, 2,
                                  QtWidgets.QTableWidgetItem('Reference'))
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

    def set_gt(self, path):
        """Set GT ComboBox items."""
        self.cb_gt.clear()
        if os.path.isfile(path):
            self.cb_gt.addItem(path)
        else:
            files = sorted(os.listdir(path))
            for f in files:
                self.cb_gt.addItem(path + '/' + f)

    def change_gt(self):
        if self.isShow:
            self.run()

    def set_scale(self):
        """Set scale."""
        scale = self.slider_scale.value()
        self.txt_scale.setText(f'{scale} %')
        self.scale = scale
        if self.isShow:
            rect = None
            if self.rect:
                rect = [
                    int(r * scale / 100.0 / self.old_scale) for r in self.rect
                ]
            self.run(rect)

    def set_column(self):
        """Set column."""
        if self.isShow:
            self.run()

    def run(self, rect=None):
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
            if self.tb_files.item(r, 2).text() != 'Reference':
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
                               self.cb_gt.currentText(), self.scale / 100.0,
                               rect)
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
        if len(path) <= 0:
            return
        if self.images:
            self.images.grab().save(path)
            QtWidgets.QMessageBox.about(self, 'Message', 'Success!')
        else:
            QtWidgets.QMessageBox.about(self, 'Message', 'Nothing to save.')

    def wheelEvent(self, ev) -> None:
        key = QtWidgets.QApplication.keyboardModifiers()
        if key == QtCore.Qt.ControlModifier:
            scale = ev.angleDelta().y() / 120
            self.scale += scale
            if self.scale < 1:
                self.scale = 1
            self.txt_scale.setText(f'{self.scale} %')
            self.slider_scale.setValue(self.scale)
            if self.scale > 200 and self.isShow:
                rect = None
                if self.rect:
                    rect = [
                        int(r * self.scale / 100.0 / self.old_scale)
                        for r in self.rect
                    ]
                self.run(rect)
        return super().wheelEvent(ev)

    def keyPressEvent(self, ev) -> None:
        if ev.key() == QtCore.Qt.Key_Left:
            if self.cb_gt.currentIndex() > 0:
                self.cb_gt.setCurrentIndex(self.cb_gt.currentIndex() - 1)
            else:
                self.cb_gt.setCurrentIndex(self.cb_gt.count() - 1)
        elif ev.key() == QtCore.Qt.Key_Right:
            if self.cb_gt.currentIndex() < self.cb_gt.count() - 1:
                self.cb_gt.setCurrentIndex(self.cb_gt.currentIndex() + 1)
            else:
                self.cb_gt.setCurrentIndex(0)
        return super().keyPressEvent(ev)

    def eventFilter(self, object, event) -> bool:
        if object == self.image_scroll:
            if event.type() == QtCore.QEvent.KeyPress:
                self.keyPressEvent(event)
                return False
        return super().eventFilter(object, event)


class SliderTab(QtWidgets.QWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.images = [None, None]
        self.imageArea = None

        # Type setting
        self.typeRect = QtWidgets.QGroupBox('Type')
        self.typeRect.setFlat(True)
        btn_type1 = QtWidgets.QRadioButton('Video', self.typeRect)
        btn_type2 = QtWidgets.QRadioButton('Image', self.typeRect)
        self.btnGroup_type = QtWidgets.QButtonGroup()
        self.btnGroup_type.addButton(btn_type1, 0)
        self.btnGroup_type.addButton(btn_type2, 1)
        self.btnGroup_type.button(0).setChecked(True)
        self.btnGroup_type.idToggled.connect(self.reset)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(btn_type1)
        hlayout.addWidget(btn_type2)
        self.typeRect.setLayout(hlayout)

        # Mode setting
        self.modeRect = QtWidgets.QGroupBox('Mode')
        self.modeRect.setFlat(True)
        btn_mode1 = QtWidgets.QRadioButton('Match', self.modeRect)
        btn_mode2 = QtWidgets.QRadioButton('Single', self.modeRect)
        self.btnGroup_mode = QtWidgets.QButtonGroup()
        self.btnGroup_mode.addButton(btn_mode1, 0)
        self.btnGroup_mode.addButton(btn_mode2, 1)
        self.btnGroup_mode.button(0).setChecked(True)
        self.btnGroup_mode.idToggled.connect(self.reset)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(btn_mode1)
        hlayout.addWidget(btn_mode2)
        self.modeRect.setLayout(hlayout)

        # Settings
        self.cb_1 = QtWidgets.QComboBox()
        self.cb_2 = QtWidgets.QComboBox()
        self.cb_1.currentIndexChanged.connect(self.change_image_1)
        self.cb_2.currentIndexChanged.connect(self.change_image_2)
        self.input_label_1 = QtWidgets.QLineEdit()
        self.input_label_2 = QtWidgets.QLineEdit()
        self.input_title = QtWidgets.QLineEdit()
        self.input_label_1.textChanged.connect(self.set_label)
        self.input_label_2.textChanged.connect(self.set_label)
        self.input_title.textChanged.connect(self.set_label)

        # Set scale
        self.scale = 100
        self.txt_scale = QtWidgets.QLabel('100 %')
        self.slider_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_scale.setMinimum(1)
        self.slider_scale.setMaximum(200)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.set_scale)

        # Auto slider setting
        self.autoRect = QtWidgets.QGroupBox('Auto Slider')
        self.autoRect.setFlat(True)
        btn_autoMode1 = QtWidgets.QRadioButton('Right', self.autoRect)
        btn_autoMode2 = QtWidgets.QRadioButton('Left', self.autoRect)
        btn_autoMode3 = QtWidgets.QRadioButton('Alternate', self.autoRect)
        self.btnGroup_auto = QtWidgets.QButtonGroup()
        self.btnGroup_auto.addButton(btn_autoMode1, 0)
        self.btnGroup_auto.addButton(btn_autoMode2, 1)
        self.btnGroup_auto.addButton(btn_autoMode3, 2)
        self.btnGroup_auto.button(2).setChecked(True)
        self.btnGroup_auto.idToggled.connect(self.set_autoSlider)
        self.slider_auto = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_auto.setMinimum(0)
        self.slider_auto.setMaximum(200)
        self.slider_auto.setValue(0)
        self.slider_auto.valueChanged.connect(self.set_autoSlider)
        glayout = QtWidgets.QGridLayout()
        glayout.addWidget(QtWidgets.QLabel('Direction:'), 0, 0, 1, 1)
        glayout.addWidget(btn_autoMode3, 0, 2, 1, 1)
        glayout.addWidget(btn_autoMode1, 0, 3, 1, 1)
        glayout.addWidget(btn_autoMode2, 0, 4, 1, 1)
        glayout.addWidget(QtWidgets.QLabel('Speed:'), 1, 0, 1, 1)
        glayout.addWidget(self.slider_auto, 1, 1, 1, 4)
        self.autoRect.setLayout(glayout)

        # Add file
        self.btn_add_1 = QtWidgets.QPushButton()
        self.btn_add_2 = QtWidgets.QPushButton()
        self.btn_add_1.setText('Add a video')
        self.btn_add_2.setText('Add frames')
        self.btn_add_1.clicked.connect(self.add_1)
        self.btn_add_2.clicked.connect(self.add_2)

        # Buttons
        self.btn_pause = QtWidgets.QPushButton()
        self.btn_pause.setText('Pause (Space)')
        self.btn_pause.clicked.connect(self.pause)
        self.btn_pause.setEnabled(False)

        self.btn_reset = QtWidgets.QPushButton()
        self.btn_reset.setText('Reset')
        self.btn_reset.clicked.connect(self.reset)

        self.btn_save = QtWidgets.QPushButton()
        self.btn_save.setText('Save')
        self.btn_save.clicked.connect(self.save)

        self.btn_record = QtWidgets.QPushButton()
        self.btn_record.setText('Record (Enter)')
        self.btn_record.clicked.connect(self.record)

        self.txt_prompt = QtWidgets.QLabel()
        self.txt_prompt.setStyleSheet('color: red; font-size: 28px')

        left_grid = QtWidgets.QGridLayout()
        left_grid.addWidget(self.typeRect, 0, 0, 1, 10)
        left_grid.addWidget(self.modeRect, 1, 0, 1, 10)
        left_grid.addWidget(QtWidgets.QLabel('Left'), 2, 0, 1, 1)
        left_grid.addWidget(self.cb_1, 2, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Right'), 3, 0, 1, 1)
        left_grid.addWidget(self.cb_2, 3, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Set label 1'), 4, 0, 1, 1)
        left_grid.addWidget(self.input_label_1, 4, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Set label 2'), 5, 0, 1, 1)
        left_grid.addWidget(self.input_label_2, 5, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Set title'), 6, 0, 1, 1)
        left_grid.addWidget(self.input_title, 6, 1, 1, 9)
        left_grid.addWidget(QtWidgets.QLabel('Set scale'), 7, 0, 1, 1)
        left_grid.addWidget(self.slider_scale, 7, 1, 1, 8)
        left_grid.addWidget(self.txt_scale, 7, 9, 1, 1)
        left_grid.addWidget(self.autoRect, 8, 0, 1, 10)
        left_grid.addWidget(self.btn_add_1, 9, 0, 1, 5)
        left_grid.addWidget(self.btn_add_2, 9, 5, 1, 5)
        left_grid.addWidget(self.btn_pause, 10, 0, 1, 10)
        left_grid.addWidget(self.btn_reset, 11, 0, 1, 10)
        left_grid.addWidget(self.btn_save, 12, 0, 1, 10)
        left_grid.addWidget(self.btn_record, 13, 0, 1, 10)
        left_grid.addWidget(QtWidgets.QLabel(), 14, 0, 10, 10)
        left_grid.addWidget(self.txt_prompt, 25, 0, 20, 10)

        # Image area
        self.image_scroll = QtWidgets.QScrollArea()
        self.image_scroll.setAlignment(QtCore.Qt.AlignCenter)
        self.image_scroll.installEventFilter(self)
        right_grid = QtWidgets.QGridLayout()
        right_grid.addWidget(self.image_scroll, 0, 0)

        # Splitter
        hsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        hsplitter.addWidget(layout2widget(left_grid))
        hsplitter.addWidget(layout2widget(right_grid))
        hsplitter.setStretchFactor(0, 1)
        hsplitter.setStretchFactor(1, 5)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(hsplitter)
        self.setLayout(hlayout)

        # Timer
        self.timer_slider = QtCore.QTimer(self)
        self.timer_slider.timeout.connect(self.auto_slider)
        self.timer_record = QtCore.QTimer(self)
        self.timer_record.timeout.connect(self.recording)

        # Player
        self.player = VideoSlider(self)
        self.player.sigout.connect(self.setImg)
        # self.player1_end = False
        # self.player2_end = False
        # self.player1 = VideoPlayer(self)
        # self.player2 = VideoPlayer(self)
        # self.player1.sigout.connect(self.setImg1)
        # self.player2.sigout.connect(self.setImg2)
        # self.player1.sigend.connect(self.set_player1)
        # self.player2.sigend.connect(self.set_player2)
        self.record_num = 0
        self.show_image()

    def set_autoSlider(self):
        if self.imageArea:
            self.timer_slider.stop()
            if self.slider_auto.value() > 0:
                self.imageArea.set_auoMode(self.btnGroup_auto.checkedId() + 1)
                self.timer_slider.start(1000 / (self.slider_auto.value()))
            else:
                self.imageArea.set_auoMode(0)

    def auto_slider(self):
        self.imageArea.auto_slider()

    def add_image(self, cb, btn):
        if self.btnGroup_mode.checkedId() == 0:
            path = QtWidgets.QFileDialog.getExistingDirectory(self)
            if len(path) <= 0:
                return
            cb.clear()
            files = sorted(os.listdir(path))
            for f in files:
                cb.addItem(path + '/' + f)
        else:
            if btn == 'add_1':
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, 'Select gt file', '', 'Images (*.jpg *.png)')
                if len(path) <= 0:
                    return
                files = [path]
            elif btn == 'add_2':
                paths = QtWidgets.QFileDialog.getExistingDirectory(self)
                if len(paths) <= 0:
                    return
                files = sorted(os.listdir(paths))
                files = [paths + '/' + f for f in files]

            for path in files:
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

    def add_1(self):
        if self.btnGroup_type.checkedId() == 1:
            self.add_image(self.cb_1, 'add_1')
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Select gt file', '', 'Images (*.mp4 *.avi)')
            if len(path) <= 0:
                return
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

    def add_2(self):
        if self.btnGroup_type.checkedId() == 1:
            self.add_image(self.cb_2, 'add_2')
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self)
            if len(path) <= 0:
                return
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

    def set_label(self):
        if self.imageArea is not None:
            self.imageArea.label_1 = self.input_label_1.text()
            self.imageArea.label_2 = self.input_label_2.text()
            self.imageArea.title = self.input_title.text()
            self.imageArea.update()

    def set_scale(self):
        """Set scale."""
        self.scale = self.slider_scale.value()
        self.txt_scale.setText(f'{self.scale} %')
        if self.imageArea is not None:
            self.imageArea.set_scale(self.scale / 100.0)

    def change_image_1(self):
        if self.btnGroup_type.checkedId() == 0:
            self.change_video()
        else:
            if self.btnGroup_mode.checkedId() == 0:
                self.cb_2.setCurrentIndex(self.cb_1.currentIndex())
            self.images[0] = cv2.imread(self.cb_1.currentText())
            self.images[1] = cv2.imread(self.cb_2.currentText())
            self.show_image()

    def change_image_2(self):
        if self.btnGroup_type.checkedId() == 0:
            self.change_video()
        else:
            if self.btnGroup_mode.checkedId() == 0:
                self.cb_1.setCurrentIndex(self.cb_2.currentIndex())
            self.images[0] = cv2.imread(self.cb_1.currentText())
            self.images[1] = cv2.imread(self.cb_2.currentText())
            self.show_image()

    def change_video(self):
        if self.cb_1.currentText() != '' and self.cb_2.currentText() != '':
            self.player.set(self.cb_1.currentText(), self.cb_2.currentText())
            self.player.start()
            self.btn_pause.setEnabled(True)

    # def setImg1(self, img):
    #     self.images[0] = img
    #     self.imageArea.setImage(self.images)
    # def setImg2(self, img):
    #     self.images[1] = img
    #     self.imageArea.setImage(self.images)
    def setImg(self, images):
        self.images = images
        self.imageArea.setImage(self.images)

    def show_image(self):
        self.imageArea = QLabelSlider(self,
                                      self.slider_scale.value() / 100.0 + 1e-7,
                                      self.input_label_1.text(),
                                      self.input_label_2.text(),
                                      self.input_title.text())
        self.imageArea.setFrameShape(QtWidgets.QFrame.Box)
        self.imageArea.setLineWidth(2)
        self.imageArea.setAlignment(QtCore.Qt.AlignBottom)
        self.imageArea.setStyleSheet(
            'border-width: 0px; border-style: solid; border-color: rgb(100, 100, 100);background-color: rgb(255, 255, 255)'  # noqa
        )
        self.image_scroll.setWidget(self.imageArea)

    def pause(self):
        if self.btn_pause.text() == 'Pause (Space)':
            self.player.pause()
            self.timer_slider.stop()
            self.btn_pause.setText('Play (Space)')
        else:
            self.player.resume()
            self.set_autoSlider()
            self.btn_pause.setText('Pause (Space)')

    def reset(self):
        self.images = [None, None]
        self.cb_1.clear()
        self.cb_2.clear()
        self.show_image()
        if self.btnGroup_type.checkedId() == 1:
            if self.btnGroup_mode.checkedId() == 0:
                self.btn_add_1.setText('Set image 1')
                self.btn_add_2.setText('Set image 2')
            else:
                self.btn_add_1.setText('Add file')
                self.btn_add_2.setText('Add directory')
        else:
            self.btn_add_1.setText('Add a video')
            self.btn_add_2.setText('Add frames')

        self.btn_pause.setText('Pause (Space)')
        self.btn_pause.setEnabled(False)
        self.btn_record.setText('Record (Enter)')

        self.timer_slider.stop()
        self.timer_record.stop()

    def save(self):
        """Save slider compare result."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'save')
        if len(path) <= 0:
            return
        if self.imageArea:
            self.imageArea.grab().save(path)
            QtWidgets.QMessageBox.about(self, 'Message', 'Success!')
        else:
            QtWidgets.QMessageBox.about(self, 'Message', 'Nothing to save.')

    def record(self):
        if self.btn_record.text() == 'Record (Enter)':
            self.record_num = 0
            self.timer_record.start(1000 / 25)
            self.txt_prompt.setText('Recording...')
            self.btn_record.setText('End (Enter)')
        elif self.btn_record.text() == 'End (Enter)':
            paths = sorted(os.listdir('.tmp/'))
            paths = ['.tmp/' + p for p in paths]
            if len(paths) <= 0:
                return
            img = cv2.imread(paths[0])
            h, w, _ = img.shape
            cur = datetime.datetime.now()
            fname = f'{cur.year}{cur.month}{cur.day}{cur.hour}{cur.minute}{cur.second}.mp4'  # noqa
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.recorder = cv2.VideoWriter(fname, fourcc, 25, (w, h))
            self.txt_prompt.setText('Saving...')
            for i in range(self.record_num):
                img = cv2.imread(paths[i])
                img = cv2.resize(img, (w, h))
                self.recorder.write(img)
                os.remove(paths[i])
            self.recorder.release()
            self.timer_record.stop()
            self.txt_prompt.setText('')
            self.btn_record.setText('Record (Enter)')
            QtWidgets.QMessageBox.about(self, 'Message',
                                        f'Save {fname} success!')

    def recording(self):
        if not os.path.isdir('.tmp/'):
            os.makedirs('.tmp/')
        fname = '.tmp/' + str(self.record_num).zfill(8) + '.png'
        self.record_num += 1
        self.imageArea.grab().save(fname)

    def wheelEvent(self, ev) -> None:
        key = QtWidgets.QApplication.keyboardModifiers()
        if key == QtCore.Qt.ControlModifier:
            scale = ev.angleDelta().y() / 120
            self.scale += scale
            if self.scale < 1:
                self.scale = 1
            self.txt_scale.setText(f'{self.scale} %')
            if self.scale > 200 and self.imageArea is not None:
                self.imageArea.set_scale(self.scale / 100.0)
            else:
                self.slider_scale.setValue(self.scale)
        return super().wheelEvent(ev)

    def keyPressEvent(self, ev) -> None:
        if ev.key() == QtCore.Qt.Key_Left:
            if self.cb_1.currentIndex() > 0:
                self.cb_1.setCurrentIndex(self.cb_1.currentIndex() - 1)
            else:
                self.cb_1.setCurrentIndex(self.cb_1.count() - 1)
        elif ev.key() == QtCore.Qt.Key_Right:
            if self.cb_1.currentIndex() < self.cb_1.count() - 1:
                self.cb_1.setCurrentIndex(self.cb_1.currentIndex() + 1)
            else:
                self.cb_1.setCurrentIndex(0)
        elif ev.key(
        ) == QtCore.Qt.Key_Enter or ev.key() + 1 == QtCore.Qt.Key_Enter:
            self.record()
        elif ev.key() == QtCore.Qt.Key_Space:
            self.pause()

    def eventFilter(self, object, event) -> bool:
        if object == self.image_scroll or object == self:
            if event.type() == QtCore.QEvent.KeyPress:
                self.keyPressEvent(event)
                return True
        return super().eventFilter(object, event)


class SRPage(QtWidgets.QWidget):

    def __init__(self, parent) -> None:
        super().__init__()

        self.parent = parent
        self.statusBar = self.parent.statusBar

        self.tab_patch = PatchTab(self)
        self.tab_slider = SliderTab(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_patch, 'patch compare')
        self.tabs.addTab(self.tab_slider, 'before/after slider')

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.tabs)
        # self.tabs.currentChanged.connect(self.tabsCurrentChanged)
