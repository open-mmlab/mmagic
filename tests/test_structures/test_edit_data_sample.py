# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import LabelData
from mmengine.testing import assert_allclose

from mmedit.structures import EditDataSample


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

    def _check_in_and_same(self, data_sample, field, value, is_meta=False):
        if is_meta:
            self.assertIn(field, data_sample.metainfo)
        else:
            self.assertIn(field, data_sample)

        if is_meta:
            val_in_data = data_sample.metainfo[field]
        else:
            val_in_data = getattr(data_sample, field)

        if isinstance(value, str):
            self.assertEqual(val_in_data, value)
        else:
            assert_allclose(val_in_data, value)

    def test_set_prefined_data(self):
        """Test fields mapping in this unit test."""
        # DATA
        gt, gt_label = torch.randn(3, 256, 256), torch.randint(0, 2, (1, ))
        fg, bg = torch.randn(3, 256, 256), torch.randn(3, 256, 256)
        alpha = torch.randn(3, 256, 256)
        ref, ref_lq = torch.randn(3, 256, 256), torch.randn(3, 256, 256)

        # METAINFO
        img_path, gt_path, merged_path = 'aaa', 'bbb', 'ccc'
        gt_channel_order, gt_color_type = 'rgb', 'color'

        data = dict(
            gt=gt,
            gt_label=gt_label,
            fg=fg,
            bg=bg,
            alpha=alpha,
            ref=ref,
            ref_lq=ref_lq,
            img_path=img_path,
            gt_path=gt_path,
            merged_path=merged_path,
            gt_channel_order=gt_channel_order,
            gt_color_type=gt_color_type)
        data_sample = EditDataSample()
        data_sample.set_predefined_data(data)

        self._check_in_and_same(data_sample, 'gt_img', gt)
        self._check_in_and_same(data_sample, 'gt_fg', fg)
        self._check_in_and_same(data_sample, 'gt_bg', bg)
        self._check_in_and_same(data_sample, 'gt_alpha', alpha)
        self._check_in_and_same(data_sample, 'ref_img', ref)
        self._check_in_and_same(data_sample, 'ref_lq', ref_lq)
        self._check_in_and_same(data_sample, 'img_path', img_path, True)
        self._check_in_and_same(data_sample, 'gt_path', gt_path, True)
        self._check_in_and_same(data_sample, 'merged_path', merged_path, True)
        self._check_in_and_same(data_sample, 'gt_channel_order',
                                gt_channel_order, True)
        self._check_in_and_same(data_sample, 'gt_color_type', gt_color_type,
                                True)
        # check gt label
        data_sample.gt_label.data = gt_label

    def _test_set_label(self, key):
        data_sample = EditDataSample()
        method = getattr(data_sample, 'set_' + key)
        # Test number
        method(1)
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test tensor with single number
        method(torch.tensor(2))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test array with single number
        method(np.array(3))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test tensor
        method(torch.tensor([1, 2, 3]))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.Tensor)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test array
        method(np.array([1, 2, 3]))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test Sequence
        method([1, 2, 3])
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test Sequence with float number
        method([0.2, 0, 0.8])
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertTrue((label.label == torch.tensor([0.2, 0, 0.8])).all())

        # Test unavailable type
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            method('hi')

        # Test set num_classes
        data_sample = EditDataSample(metainfo={'num_classes': 10})
        method = getattr(data_sample, 'set_' + key)
        method(5)
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIn('num_classes', label)
        self.assertEqual(label.num_classes, 10)

        # Test unavailable label
        with self.assertRaisesRegex(ValueError, r'data .*[15].* should '):
            method(15)

    def test_set_gt_label(self):
        self._test_set_label('gt_label')

    def test_del_gt_label(self):
        data_sample = EditDataSample()
        self.assertNotIn('gt_label', data_sample)
        data_sample.set_gt_label(1)
        self.assertIn('gt_label', data_sample)
        del data_sample.gt_label
        self.assertNotIn('gt_label', data_sample)
