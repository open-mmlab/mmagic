# Copyright (c) OpenMMLab. All rights reserved.
from PyQt5 import QtCore, QtGui, QtWidgets

from tools.gui.utils import layout2widget


class QLabelClick(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event):
        self.clicked.emit(self.text())


class ConcatImageWidget(QtWidgets.QWidget):

    def __init__(self, parent=None, dir='./', col_num=4):
        super(ConcatImageWidget, self).__init__(parent)
        self.col_num = col_num
        self.hlayout = QtWidgets.QHBoxLayout()
        self.vlayout = QtWidgets.QVBoxLayout()

    def set_images(self, file_path, labels, gt=None):
        for i in reversed(range(self.vlayout.count())):
            self.vlayout.itemAt(i).widget().deleteLater()
        for i in reversed(range(self.hlayout.count())):
            self.hlayout.itemAt(i).widget().deleteLater()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout_text = QtWidgets.QHBoxLayout()
        hlayout_text.setContentsMargins(0, 0, 0, 0)
        total_h = 0
        for i, (path, text) in enumerate(zip(file_path, labels)):
            img = QtGui.QPixmap(path)
            label = QtWidgets.QLabel()
            label.setMargin(0)
            label.setPixmap(img)
            hlayout.addWidget(label)
            label_text = QtWidgets.QLabel()
            label_text.setMargin(0)
            label_text.setAlignment(QtCore.Qt.AlignCenter)
            label_text.setText(text)
            label_text.adjustSize()
            hlayout_text.addWidget(label_text)

            if (i + 1) % self.col_num == 0:
                self.vlayout.addWidget(layout2widget(hlayout))
                self.vlayout.addWidget(layout2widget(hlayout_text))

                hlayout = QtWidgets.QHBoxLayout()
                hlayout.setContentsMargins(0, 0, 0, 0)
                hlayout_text = QtWidgets.QHBoxLayout()
                hlayout_text.setContentsMargins(0, 0, 0, 0)

                total_h += label.height() + label_text.height() + 12

        if len(hlayout) > 0:
            for i in range(0, 4 - len(hlayout)):
                label = QtWidgets.QLabel()
                label.setMargin(0)
                hlayout.addWidget(label)
                label = QtWidgets.QLabel()
                label.setMargin(0)
                hlayout_text.addWidget(label)
            self.vlayout.addWidget(layout2widget(hlayout))
            self.vlayout.addWidget(layout2widget(hlayout_text))
            total_h += label.height()
        else:
            total_h -= label_text.height() + 12

        if gt:
            img = QtGui.QPixmap(gt)
            w, h = img.width(), img.height()
            new_h = total_h
            new_w = int(new_h / h * w)
            img = img.scaled(new_w, new_h)
            label = QtWidgets.QLabel()
            label.setMargin(0)
            label.setAlignment(QtCore.Qt.AlignTop)
            label.setPixmap(img)
            self.hlayout.addWidget(label)
            self.hlayout.addWidget(layout2widget(self.vlayout))
            self.setLayout(self.hlayout)

        else:
            self.setLayout(self.vlayout)
