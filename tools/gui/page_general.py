# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from utils import layout2widget


class SliderTab(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.images = [None, None]

        self.cb_1 = QtWidgets.QComboBox()
        self.cb_2 = QtWidgets.QComboBox()
        self.cb_1.currentIndexChanged.connect(self.change_image)
        self.cb_2.currentIndexChanged.connect(self.change_image)
        self.btn_add = QtWidgets.QPushButton()
        self.btn_add.setText('Add')
        self.btn_add.clicked.connect(self.add)
        left_grid = QtWidgets.QGridLayout()
        left_grid.addWidget(self.cb_1, 0, 0)
        left_grid.addWidget(self.cb_2, 1, 0)
        left_grid.addWidget(self.btn_add, 2, 0)

        self.imageArea = QtWidgets.QLabel()
        self.imageArea.setFrameShape(QtWidgets.QFrame.Box)
        self.imageArea.setLineWidth(2)
        self.imageArea.setAlignment(QtCore.Qt.AlignBottom)
        self.imageArea.setStyleSheet(
            'border-width: 0px; border-style: solid; border-color: rgb(100, 100, 100);background-color: rgb(255, 255, 255)'  # noqa
        )
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMaximum(800)
        self.slider.valueChanged.connect(self.show_image)
        right_grid = QtWidgets.QGridLayout()
        right_grid.addWidget(self.imageArea, 0, 0)
        right_grid.addWidget(self.slider, 1, 0)

        # Splitter
        hsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        hsplitter.addWidget(layout2widget(left_grid))
        hsplitter.addWidget(layout2widget(right_grid))
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(hsplitter)
        self.setLayout(hlayout)

    def add(self):
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
        if self.images[0] is None or self.images[1] is None:
            return
        self.show_image()

    def show_image(self):
        img1, img2 = self.images
        h2, w2, c2 = img2.shape
        img2 = cv2.resize(img2, (800, int(800 / w2 * h2)))
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
        jpg = QtGui.QPixmap.fromImage(img_dis).scaled(
            self.imageArea.width(), int(self.imageArea.width() / w2 * h2))
        self.imageArea.setPixmap(jpg)


class GeneralPage(QtWidgets.QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.tab_slider = SliderTab()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_slider, 'before/after slider')

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.tabs)
