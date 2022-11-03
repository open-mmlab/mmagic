# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from PyQt5 import QtGui, QtWidgets


def layout2widget(layout):
    wg = QtWidgets.QWidget()
    wg.setLayout(layout)
    return wg


def qimage2array(img):
    w = img.width()
    h = img.height()
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    img = img.bits()
    img.setsize(w * h * 4)
    img = np.frombuffer(img, np.uint8)
    img = np.reshape(img, (h, w, 4))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img
