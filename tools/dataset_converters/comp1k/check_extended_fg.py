#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
# This script checks the alpha-foreground difference between
# the extended fg and the original fg

import glob
import os
import os.path as osp

import cv2
import numpy as np

folder = 'data/adobe_composition-1k/Training_set/Adobe-licensed images'
folder = osp.join(folder.split('/'))
imgs = [
    os.path.splitext(os.path.basename(x))[0]
    for x in glob.glob(osp.join(folder, 'fg', '*.jpg'))
]

print('max,avg,img')
for name in imgs:
    alpha = cv2.imread(
        osp.join(folder, 'alpha', '*.jpg'), cv2.IMREAD_GRAYSCALE).astype(
            np.float32)[..., None] / 255
    fg = cv2.imread(osp.join(folder, 'fg', f'{name}.jpg')).astype(np.float32)
    xt = cv2.imread(osp.join(folder, 'fg_extended',
                             f'{name}.jpg')).astype(np.float32)
    diff = np.abs((fg - xt) * alpha)
    print(f'{diff.max()},{diff.mean()},"{name}"', flush=True)
