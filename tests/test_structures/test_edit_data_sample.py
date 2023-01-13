# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine.structures import BaseDataElement

from mmedit.structures import EditDataSample, PixelData


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestEditDataSample(TestCase):

    def test_init(self):
        meta_info = dict(
            target_size=[256, 256],
            scale_factor=np.array([1.5, 1.5]),
            img_shape=torch.rand(4))

        edit_data_sample = EditDataSample(metainfo=meta_info)
        assert 'target_size' in edit_data_sample
        assert edit_data_sample.target_size == [256, 256]
        assert edit_data_sample.get('target_size') == [256, 256]

    def test_setter(self):

        edit_data_sample = EditDataSample()

        # test gt_img
        gt_img_data = dict(
            metainfo=dict(path='gt.py'),
            img=np.random.randint(0, 255, (3, 256, 256)),
            img1=np.random.randint(0, 255, (3, 256, 256)))
        gt_img = PixelData(**gt_img_data)
        edit_data_sample.gt_img = gt_img
        assert 'gt_img' in edit_data_sample
        assert _equal(edit_data_sample.gt_img.img, gt_img_data['img'])
        assert _equal(edit_data_sample.gt_img.img1, gt_img_data['img1'])

        # test frames
        gt_img_data = dict(
            metainfo=dict(path='gt.py'),
            img=np.random.randint(0, 255, (10, 3, 256, 256)))
        gt_img = PixelData(**gt_img_data)
        edit_data_sample.gt_img = gt_img
        assert 'gt_img' in edit_data_sample
        assert _equal(edit_data_sample.gt_img.img, gt_img_data['img'])

        # test img_lq
        img_lq_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        img_lq = PixelData(**img_lq_data)
        edit_data_sample.img_lq = img_lq
        assert 'img_lq' in edit_data_sample
        assert _equal(edit_data_sample.img_lq.img, img_lq_data['img'])

        # test pred_img
        pred_img_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        pred_img = PixelData(**pred_img_data)
        edit_data_sample.pred_img = pred_img
        assert 'pred_img' in edit_data_sample
        assert _equal(edit_data_sample.pred_img.img, pred_img_data['img'])

        # test ref_img
        ref_img_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        ref_img = PixelData(**ref_img_data)
        edit_data_sample.ref_img = ref_img
        assert 'ref_img' in edit_data_sample
        assert _equal(edit_data_sample.ref_img.img, ref_img_data['img'])

        # test ref_lq
        ref_lq_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        ref_lq = PixelData(**ref_lq_data)
        edit_data_sample.ref_lq = ref_lq
        assert 'ref_lq' in edit_data_sample
        assert _equal(edit_data_sample.ref_lq.img, ref_lq_data['img'])

        # test mask
        mask_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        mask = PixelData(**mask_data)
        edit_data_sample.mask = mask
        assert 'mask' in edit_data_sample
        assert _equal(edit_data_sample.mask.img, mask_data['img'])

        # test gt_unsharp
        gt_unsharp_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        gt_unsharp = PixelData(**gt_unsharp_data)
        edit_data_sample.gt_unsharp = gt_unsharp
        assert 'gt_unsharp' in edit_data_sample
        assert _equal(edit_data_sample.gt_unsharp.img, gt_unsharp_data['img'])

        # test gt_heatmap
        gt_heatmap_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        gt_heatmap = PixelData(**gt_heatmap_data)
        edit_data_sample.gt_heatmap = gt_heatmap
        assert 'gt_heatmap' in edit_data_sample
        assert _equal(edit_data_sample.gt_heatmap.img, gt_heatmap_data['img'])

        # test pred_heatmap
        pred_heatmap_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        pred_heatmap = PixelData(**pred_heatmap_data)
        edit_data_sample.pred_heatmap = pred_heatmap
        assert 'pred_heatmap' in edit_data_sample
        assert _equal(edit_data_sample.pred_heatmap.img,
                      pred_heatmap_data['img'])

        # test trimap
        trimap_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        trimap = PixelData(**trimap_data)
        edit_data_sample.trimap = trimap
        assert 'trimap' in edit_data_sample
        assert _equal(edit_data_sample.trimap.img, trimap_data['img'])

        # test gt_alpha
        gt_alpha_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        gt_alpha = PixelData(**gt_alpha_data)
        edit_data_sample.gt_alpha = gt_alpha
        assert 'gt_alpha' in edit_data_sample
        assert _equal(edit_data_sample.gt_alpha.img, gt_alpha_data['img'])

        # test pred_alpha
        pred_alpha_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        pred_alpha = PixelData(**pred_alpha_data)
        edit_data_sample.pred_alpha = pred_alpha
        assert 'pred_alpha' in edit_data_sample
        assert _equal(edit_data_sample.pred_alpha.img, pred_alpha_data['img'])

        # test gt_fg
        gt_fg_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        gt_fg = PixelData(**gt_fg_data)
        edit_data_sample.gt_fg = gt_fg
        assert 'gt_fg' in edit_data_sample
        assert _equal(edit_data_sample.gt_fg.img, gt_fg_data['img'])

        # test pred_fg
        pred_fg_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        pred_fg = PixelData(**pred_fg_data)
        edit_data_sample.pred_fg = pred_fg
        assert 'pred_fg' in edit_data_sample
        assert _equal(edit_data_sample.pred_fg.img, pred_fg_data['img'])

        # test gt_bg
        gt_bg_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        gt_bg = PixelData(**gt_bg_data)
        edit_data_sample.gt_bg = gt_bg
        assert 'gt_bg' in edit_data_sample
        assert _equal(edit_data_sample.gt_bg.img, gt_bg_data['img'])

        # test pred_bg
        pred_bg_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        pred_bg = PixelData(**pred_bg_data)
        edit_data_sample.pred_bg = pred_bg
        assert 'pred_bg' in edit_data_sample
        assert _equal(edit_data_sample.pred_bg.img, pred_bg_data['img'])

        # test gt_merged
        ignored_data_data = dict(bboxes=torch.rand(4, 4), labels=torch.rand(4))
        ignored_data = BaseDataElement(**ignored_data_data)
        edit_data_sample.ignored_data = ignored_data
        assert 'ignored_data' in edit_data_sample
        assert _equal(edit_data_sample.ignored_data.bboxes,
                      ignored_data_data['bboxes'])
        assert _equal(edit_data_sample.ignored_data.labels,
                      ignored_data_data['labels'])

        # test shape error
        with pytest.raises(AssertionError):
            gt_img_data = dict(
                metainfo=dict(path='gt.py'),
                img=np.random.randint(0, 255, (3, 256, 256)),
                img1=np.random.randint(0, 255, (3, 256, 257)))
            gt_img = PixelData(**gt_img_data)

    def test_deleter(self):
        img_data = dict(img=np.random.randint(0, 255, (3, 256, 256)))
        edit_data_sample = EditDataSample()

        gt_img = PixelData(**img_data)
        edit_data_sample.gt_img = gt_img
        assert 'gt_img' in edit_data_sample
        del edit_data_sample.gt_img
        assert 'gt_img' not in edit_data_sample

        pred_img = PixelData(**img_data)
        edit_data_sample.pred_img = pred_img
        assert 'pred_img' in edit_data_sample
        del edit_data_sample.pred_img
        assert 'pred_img' not in edit_data_sample

        ref_img = PixelData(**img_data)
        edit_data_sample.ref_img = ref_img
        assert 'ref_img' in edit_data_sample
        del edit_data_sample.ref_img
        assert 'ref_img' not in edit_data_sample

        mask = PixelData(**img_data)
        edit_data_sample.mask = mask
        assert 'mask' in edit_data_sample
        del edit_data_sample.mask
        assert 'mask' not in edit_data_sample

        gt_heatmap = PixelData(**img_data)
        edit_data_sample.gt_heatmap = gt_heatmap
        assert 'gt_heatmap' in edit_data_sample
        del edit_data_sample.gt_heatmap
        assert 'gt_heatmap' not in edit_data_sample

        pred_heatmap = PixelData(**img_data)
        edit_data_sample.pred_heatmap = pred_heatmap
        assert 'pred_heatmap' in edit_data_sample
        del edit_data_sample.pred_heatmap
        assert 'pred_heatmap' not in edit_data_sample

        trimap = PixelData(**img_data)
        edit_data_sample.trimap = trimap
        assert 'trimap' in edit_data_sample
        del edit_data_sample.trimap
        assert 'trimap' not in edit_data_sample

        gt_alpha = PixelData(**img_data)
        edit_data_sample.gt_alpha = gt_alpha
        assert 'gt_alpha' in edit_data_sample
        del edit_data_sample.gt_alpha
        assert 'gt_alpha' not in edit_data_sample

        pred_alpha = PixelData(**img_data)
        edit_data_sample.pred_alpha = pred_alpha
        assert 'pred_alpha' in edit_data_sample
        del edit_data_sample.pred_alpha
        assert 'pred_alpha' not in edit_data_sample

        gt_fg = PixelData(**img_data)
        edit_data_sample.gt_fg = gt_fg
        assert 'gt_fg' in edit_data_sample
        del edit_data_sample.gt_fg
        assert 'gt_fg' not in edit_data_sample

        pred_fg = PixelData(**img_data)
        edit_data_sample.pred_fg = pred_fg
        assert 'pred_fg' in edit_data_sample
        del edit_data_sample.pred_fg
        assert 'pred_fg' not in edit_data_sample

        gt_bg = PixelData(**img_data)
        edit_data_sample.gt_bg = gt_bg
        assert 'gt_bg' in edit_data_sample
        del edit_data_sample.gt_bg
        assert 'gt_bg' not in edit_data_sample

        pred_bg = PixelData(**img_data)
        edit_data_sample.pred_bg = pred_bg
        assert 'pred_bg' in edit_data_sample
        del edit_data_sample.pred_bg
        assert 'pred_bg' not in edit_data_sample

        gt_merged = PixelData(**img_data)
        edit_data_sample.gt_merged = gt_merged
        assert 'gt_merged' in edit_data_sample
        del edit_data_sample.gt_merged
        assert 'gt_merged' not in edit_data_sample
