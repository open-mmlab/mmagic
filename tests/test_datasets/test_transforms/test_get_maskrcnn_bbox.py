# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2 as cv

from mmedit.datasets.transforms import InstanceCrop
from mmedit.utils import tensor2img


class TestMaskRCNNBbox:

    DEFAULT_ARGS = dict(key='img', finesize=256)

    def test_maskrcnn_bbox(self):
        detectetor = InstanceCrop(**self.DEFAULT_ARGS, stage='test')
        data_root = '..'
        img_path = 'data/image/gray/test.jpg'
        img = cv.imread(os.path.join(data_root, img_path))

        data = dict(img=img)

        results = detectetor(data)
        pred_bbox = results.pred_bbox

        assert len(pred_bbox) <= 8
        assert results['full_gray'] and results['box_info'] \
               and results['cropped_gray']

        detectetor.stage = 'fusion'
        results = detectetor(data)
        index = len(results.pred_bbox)
        assert results['full_rgb'] and results['cropped_rgb']
        assert results['cropped_gray_list'].shape == (index, 3, 256, 256)

        detectetor.stage = 'full'
        results = detectetor(data)
        assert results['rgb_img'] and results['gray_img']
        assert tensor2img(results['rgb_img']).shape == (3, 256, 256)

    def test_gen_maskrcnn_from_pred(self):
        detectetor = InstanceCrop(**self.DEFAULT_ARGS, stage='test')
        data_root = '..'
        img_path = 'data/image/gray/test.jpg'
        img = cv.imread(os.path.join(data_root, img_path))

        box_num_upbound = 4
        pred_bbox = detectetor.gen_maskrcnn_bbox_fromPred(img)

        assert len(pred_bbox) <= box_num_upbound
        assert pred_bbox.shape[-1] == 4

    def test_get_box_info(self):
        detectetor = InstanceCrop(**self.DEFAULT_ARGS, stage='test')
        data_root = '..'
        img_path = 'data/image/gray/test.jpg'
        img = cv.imread(os.path.join(data_root, img_path))

        pred_bbox = detectetor.gen_maskrcnn_bbox_fromPred(img)

        resize_startx = int(pred_bbox[0] / img.shape[0] * 256)
        resize_starty = int(pred_bbox[1] / img.shape[1] * 256)
        resize_endx = int(pred_bbox[2] / img.shape[0] * 256)
        resize_endy = int(pred_bbox[3] / img.shape[1] * 256)

        box_info = detectetor.get_box_info(pred_bbox, img.shape)

        assert box_info[0] == resize_starty and \
            box_info[1] == 256 - resize_endx and \
            box_info[2] == resize_starty and \
            box_info[3] == 256 - resize_endy and \
            box_info[4] == resize_endx - resize_startx and \
            box_info[5] == resize_endy - resize_starty
