# Copyright (c) OpenMMLab. All rights reserved.
from PyQt5 import QtWidgets


def layout2widget(layout):
    wg = QtWidgets.QWidget()
    wg.setLayout(layout)
    return wg
