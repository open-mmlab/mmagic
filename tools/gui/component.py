# Copyright (c) OpenMMLab. All rights reserved.
from PyQt5 import QtCore, QtGui, QtWidgets
from utils import layout2widget


class QLabelClick(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event):
        self.clicked.emit(self.text())


class QLabelPaint(QtWidgets.QLabel):

    def __init__(self, parent):
        super().__init__()
        self.beginPoint = None
        self.endPoint = None
        self.parent = parent
        self.statusBar = self.parent.statusBar
        self.isShow = False

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            self.endPoint = None
            self.beginPoint = ev.pos()
            self.isShow = False
        return super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            self.endPoint = ev.pos()
            self.update()
            self.isShow = True
        return super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.endPoint = ev.pos()
        self.update()
        self.statusBar.showMessage(
            f'Start: {self.beginPoint.x()},{self.beginPoint.y()}; \
                End: {self.endPoint.x()},{self.endPoint.y()}')
        return super().mouseMoveEvent(ev)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        super().paintEvent(ev)
        if self.beginPoint and self.endPoint:
            qp = QtGui.QPainter()
            qp.begin(self)
            pen = QtGui.QPen(QtCore.Qt.red, 2)
            qp.setPen(pen)
            w = abs(self.beginPoint.x() - self.endPoint.x())
            h = abs(self.beginPoint.y() - self.endPoint.y())
            qp.drawRect(self.beginPoint.x(), self.beginPoint.y(), w, h)
            qp.end()
            if self.isShow and isinstance(self.parent, ConcatImageWidget):
                self.parent.show_images(self.beginPoint.x(),
                                        self.beginPoint.y(), w, h)
                self.isShow = False


class ConcatImageWidget(QtWidgets.QWidget):

    def __init__(self, parent, mode, col_num=4):
        super(ConcatImageWidget, self).__init__(parent)
        self.parent = parent
        self.statusBar = self.parent.statusBar
        self.hlayout = QtWidgets.QHBoxLayout()
        self.setLayout(self.hlayout)
        self.mode = mode
        self.col_num = col_num
        self.file_path = None
        self.labels = None
        self.gt = None
        self.gt_scale = 1
        self.img_h = 0

    def show_images(self, x=0, y=0, w=0, h=0):
        vlayout = QtWidgets.QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        hlayout_img = QtWidgets.QHBoxLayout()
        hlayout_img.setContentsMargins(0, 0, 0, 0)
        hlayout_text = QtWidgets.QHBoxLayout()
        hlayout_text.setContentsMargins(0, 0, 0, 0)

        for i, (path, text) in enumerate(zip(self.file_path, self.labels)):
            img = QtGui.QPixmap(path).scaled(self.gt_w, self.gt_h)
            if self.mode == 0:
                img = img.copy(QtCore.QRect(x, y, w, h))
            if self.img_h > 0:
                img_w = int(float(self.img_h) / img.height() * img.width())
                img = img.scaled(img_w, self.img_h)

            label = QtWidgets.QLabel()
            label.setFixedWidth(img.width())
            label.setFixedHeight(img.height())
            label.setMargin(0)
            label.setPixmap(img)
            hlayout_img.addWidget(label)

            label_text = QtWidgets.QLabel()
            label_text.setMargin(0)
            label_text.setAlignment(QtCore.Qt.AlignCenter)
            label_text.setText(text)
            label_text.adjustSize()
            hlayout_text.addWidget(label_text)

            if (i + 1) % self.col_num == 0:
                vlayout.addWidget(layout2widget(hlayout_img))
                vlayout.addWidget(layout2widget(hlayout_text))
                hlayout_img = QtWidgets.QHBoxLayout()
                hlayout_img.setContentsMargins(0, 0, 0, 0)
                hlayout_text = QtWidgets.QHBoxLayout()
                hlayout_text.setContentsMargins(0, 0, 0, 0)

        if len(hlayout_img) > 0:
            for i in range(0, self.col_num - len(hlayout_img)):
                label = QtWidgets.QLabel()
                label.setMargin(0)
                label.setFixedWidth(img.width())
                label.setFixedHeight(img.height())
                hlayout_img.addWidget(label)
                label = QtWidgets.QLabel()
                label.setMargin(0)
                hlayout_text.addWidget(label)
            vlayout.addWidget(layout2widget(hlayout_img))
            vlayout.addWidget(layout2widget(hlayout_text))

        total_w = self.gt_w + (img.width() + 2) * self.col_num
        self.setFixedWidth(total_w)
        if self.hlayout.count() > 1:
            item = self.hlayout.itemAt(1)
            self.hlayout.removeItem(item)
            if item.widget():
                item.widget().deleteLater()
            self.hlayout.addWidget(layout2widget(vlayout))
        else:
            self.hlayout.addWidget(layout2widget(vlayout))

    def set_images(self, file_path, labels, gt=None, scale=1):
        self.file_path = file_path
        self.labels = labels
        self.gt = gt

        for i in reversed(range(self.hlayout.count())):
            self.hlayout.itemAt(i).widget().deleteLater()
        self.hlayout.setContentsMargins(0, 0, 0, 0)

        img = QtGui.QPixmap(self.gt)
        self.gt_w, self.gt_h = img.width() * scale, img.height() * scale
        img = img.scaled(self.gt_w, self.gt_h)
        row = (len(self.file_path) + self.col_num - 1) // self.col_num
        self.img_h = int(float(self.gt_h - (row - 1) * 29) / row)

        label = QLabelPaint(self)
        label.setMargin(0)
        label.setAlignment(QtCore.Qt.AlignTop)
        label.setPixmap(img)
        self.hlayout.addWidget(label)

        self.show_images(0, 0, self.gt_w, self.gt_h)
