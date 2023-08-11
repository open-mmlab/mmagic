# Copyright (c) OpenMMLab. All rights reserved.
import os
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from utils import layout2widget


class QLabelClick(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event):
        self.clicked.emit(self.text())


class QLabelSlider(QtWidgets.QLabel):

    def __init__(self, parent, scale, label_1, label_2, title):
        super().__init__()
        self.parent = parent
        self.hSlider = -1
        self.vSlider = -1
        self.oldSlider = -1
        self.scale = scale
        self.label_1 = label_1
        self.label_2 = label_2
        self.title = title
        self.images = self.parent.images
        self.auto_mode = 0

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            if self.hSlider > -1:
                self.oldSlider = self.hSlider
            elif self.vSlider > -1:
                self.oldSlider = self.vSlider
        return super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            self.oldSlider = -1
            self.update()
        return super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self.oldSlider > -1:
            if self.hSlider > -1:
                self.hSlider = ev.pos().x()
            elif self.vSlider > -1:
                self.vSlider = ev.pos().y()
        self.update()
        return super().mouseMoveEvent(ev)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.drawImage(0, 0, QtGui.QImage(self.getImage()))
        pen = QtGui.QPen(QtCore.Qt.green, 3)
        qp.setPen(pen)
        qp.drawLine(self.hSlider, 0, self.hSlider, self.height())
        length = 9
        qp.drawText(self.hSlider - 10 - len(self.label_1) * length, 20,
                    self.label_1)
        qp.drawText(self.hSlider + 10, 20, self.label_2)
        qp.drawText(10, self.height() - 10, self.title)
        qp.end()

    def set_scale(self, scale):
        self.hSlider = int(self.hSlider * scale / self.scale)
        self.scale = scale
        self.update()

    def setImage(self, images):
        self.images = images
        self.update()

    def set_auoMode(self, mode):
        if mode != 3 or self.auto_mode < 3:
            self.auto_mode = mode

    def auto_slider(self):
        try:
            if self.auto_mode == 1:
                self.hSlider += 1
                if self.hSlider > self.w:
                    self.hSlider = 0
            elif self.auto_mode == 2:
                self.hSlider -= 1
                if self.hSlider < 0:
                    self.hSlider = self.w
            elif self.auto_mode == 3:
                self.hSlider += 1
                if self.hSlider >= self.w:
                    self.auto_mode = 4
            elif self.auto_mode == 4:
                self.hSlider -= 1
                if self.hSlider <= 0:
                    self.auto_mode = 3
            self.update()

        except Exception:
            print(Exception)
            pass

    def getImage(self):
        img1, img2 = self.images
        if img1 is None or img2 is None:
            return
        h1, w1, c = img1.shape
        h2, w2, c = img2.shape
        if w2 > w1:
            img1 = cv2.resize(img1, (w2, h2))
            h, w = h2, w2
        else:
            img2 = cv2.resize(img2, (w1, h1))
            h, w = h1, w1
        self.h = int(h * self.scale)
        self.w = int(w * self.scale)
        self.setFixedHeight(self.h)
        self.setFixedWidth(self.w)
        if self.hSlider < 0:
            self.hSlider = int(self.w / 2.0)

        v = int(self.hSlider / self.scale)
        img11 = img1[:, 0:v].copy()
        img22 = img2[:, v:].copy()
        img = np.hstack((img11, img22))
        # img = cv2.line(img, (v, 0), (v, h2), (0, 222, 0), 4)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_dis = QtGui.QImage(rgb_img, w, h, w * c,
                               QtGui.QImage.Format_RGB888)
        img = QtGui.QPixmap.fromImage(img_dis).scaled(self.width(),
                                                      self.height())
        return img


class QLabelPaint(QtWidgets.QLabel):

    def __init__(self, parent, beginPoint=None, endPoint=None):
        super().__init__()
        self.beginPoint = beginPoint
        self.endPoint = endPoint
        self.parent = parent
        self.statusBar = self.parent.statusBar
        if self.beginPoint and self.endPoint:
            self.isShow = True
        else:
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
            self.parent.set_rect(self.beginPoint, self.endPoint)
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
        self.scale = 1
        self.col_num = col_num
        self.file_path = None
        self.labels = None
        self.gt = None
        self.img_h = 0
        self.rect = None

    def show_images(self, x=0, y=0, w=0, h=0):
        self.rect = [x, y, w, h]
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

    def set_images(self, file_path, labels, gt=None, scale=1, rect=None):
        self.file_path = file_path
        self.labels = labels
        self.gt = gt
        self.scale = scale

        for i in reversed(range(self.hlayout.count())):
            self.hlayout.itemAt(i).widget().deleteLater()
        self.hlayout.setContentsMargins(0, 0, 0, 0)

        img = QtGui.QPixmap(self.gt)
        self.gt_w, self.gt_h = img.width() * scale, img.height() * scale
        img = img.scaled(self.gt_w, self.gt_h)
        row = (len(self.file_path) + self.col_num - 1) // self.col_num
        self.img_h = int(float(self.gt_h - (row - 1) * 29) / row)

        beginPoint = None
        endPoint = None
        if rect:
            beginPoint = QtCore.QPoint(rect[0], rect[1])
            endPoint = QtCore.QPoint(rect[2], rect[3])
        label = QLabelPaint(self, beginPoint, endPoint)
        label.setMargin(0)
        label.setAlignment(QtCore.Qt.AlignTop)
        label.setPixmap(img)
        self.hlayout.addWidget(label)
        if rect:
            self.show_images(rect[0], rect[1], rect[2] - rect[0],
                             rect[3] - rect[1])
        else:
            self.show_images(0, 0, self.gt_w, self.gt_h)

    def set_rect(self, beginPoint, endPoint):
        self.parent.rect = [
            beginPoint.x(),
            beginPoint.y(),
            endPoint.x(),
            endPoint.y()
        ]
        self.parent.old_scale = self.scale


class VideoPlayer(QtCore.QThread):
    sigout = QtCore.pyqtSignal(np.ndarray)
    sigend = QtCore.pyqtSignal(bool)

    def __init__(self, parent):
        super(VideoPlayer, self).__init__(parent)
        self.parent = parent

    def set(self, path, fps=None):
        self.path = path
        self.video, self.fps, self.actual_frames = self.setVideo(path, fps)
        self.time = self.actual_frames / self.fps
        self.total_frames = self.actual_frames
        self.num = 0
        self.working = True
        self.isPause = False
        self.mutex = QtCore.QMutex()
        self.cond = QtCore.QWaitCondition()

    def setVideo(self, path, fps):
        if os.path.isfile(path):
            v = cv2.VideoCapture(path)
            total_frames = v.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps is None:
                fps = v.get(cv2.CAP_PROP_FPS)
        else:
            files = sorted(os.listdir(path))
            v = ['/'.join([path, f]) for f in files]
            total_frames = len(v)
            if fps is None:
                fps = 25
        return v, fps, total_frames

    def pause(self):
        self.isPause = True

    def resume(self):
        self.isPause = False
        self.cond.wakeAll()

    def __del__(self):
        self.working = False
        self.wait()

    def run(self):
        while self.working:
            self.mutex.lock()
            if self.isPause:
                self.cond.wait(self.mutex)
            if isinstance(self.video, list):
                img = cv2.imread(self.video[self.num])
                self.num += 1
                self.sigout.emit(img)
                if self.num >= self.total_frames:
                    self.sigend.emit(True)
                    self.num = 0
                time.sleep(1 / self.fps)
            self.mutex.unlock()


class VideoSlider(QtCore.QThread):
    sigout = QtCore.pyqtSignal(list)
    sigend = QtCore.pyqtSignal(bool)

    def __init__(self, parent):
        super(VideoSlider, self).__init__(parent)
        self.parent = parent
        self.mutex = QtCore.QMutex()
        self.cond = QtCore.QWaitCondition()

    def set(self, path1, path2, fps1=None, fps2=None):
        self.path1 = path1
        self.path2 = path2
        self.v1, self.fps1, self.total_frames1 = self.setVideo(path1, fps1)
        self.v2, self.fps2, self.total_frames2 = self.setVideo(path2, fps2)
        if self.fps1 != self.fps2:
            return False
        self.fps = self.fps1
        self.num = 0
        self.working = True
        self.isPause = False
        return True

    def setVideo(self, path, fps):
        if os.path.isfile(path):
            v = cv2.VideoCapture(path)
            total_frames = v.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = v.get(cv2.CAP_PROP_FPS)
        else:
            files = sorted(os.listdir(path))
            v = ['/'.join([path, f]) for f in files]
            total_frames = len(v)
            if fps is None:
                fps = 25
        return v, fps, total_frames

    def pause(self):
        self.isPause = True

    def resume(self):
        self.isPause = False
        self.cond.wakeAll()

    def __del__(self):
        self.working = False
        self.wait()

    def run(self):
        while self.working:
            self.mutex.lock()
            if self.isPause:
                self.cond.wait(self.mutex)
            if isinstance(self.v1, list):
                num = self.num if self.num < self.total_frames1 \
                    else self.total_frames1 - 1
                img1 = cv2.imread(self.v1[num])
            elif isinstance(self.v1, cv2.VideoCapture):
                r, img1 = self.v1.read()
                if not r:
                    self.v1, self.fps1, self.total_frames1 = self.setVideo(
                        self.path1, self.fps1)
            if isinstance(self.v2, list):
                num = self.num if self.num < self.total_frames2 \
                    else self.total_frames2 - 1
                img2 = cv2.imread(self.v2[num])
            elif isinstance(self.v2, cv2.VideoCapture):
                r, img2 = self.v2.read()
                if not r:
                    self.v2, self.fps2, self.total_frames2 = self.setVideo(
                        self.path2, self.fps2)
            self.num += 1
            self.sigout.emit([img1, img2])
            if self.num >= self.total_frames1 and \
                    self.num >= self.total_frames2:
                self.num = 0
            time.sleep(1 / self.fps)
            self.mutex.unlock()
