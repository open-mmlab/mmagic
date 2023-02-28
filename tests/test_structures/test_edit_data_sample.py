# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import LabelData
from mmengine.testing import assert_allclose

from mmedit.structures import EditDataSample
from mmedit.structures.edit_data_sample import is_splitable_var


def test_is_stacked_var():
    assert is_splitable_var(EditDataSample())
    assert is_splitable_var(torch.randn(10, 10))
    assert is_splitable_var(np.ndarray((10, 10)))
    assert is_splitable_var([1, 2])
    assert is_splitable_var((1, 2))
    assert not is_splitable_var({'a': 1})
    assert not is_splitable_var('a')


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
        assert len(edit_data_sample) == 1

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

    def test_stack_and_split(self):
        # test stack
        data_sample1 = EditDataSample()
        data_sample1.set_gt_label(1)
        data_sample1.set_tensor_data({'img': torch.randn(3, 4, 5)})
        data_sample1.set_data({'mode': 'a'})
        data_sample1.set_metainfo({
            'channel_order': 'rgb',
            'color_flag': 'color'
        })
        data_sample2 = EditDataSample()
        data_sample2.set_gt_label(2)
        data_sample2.set_tensor_data({'img': torch.randn(3, 4, 5)})
        data_sample2.set_data({'mode': 'b'})
        data_sample2.set_metainfo({
            'channel_order': 'rgb',
            'color_flag': 'color'
        })

        data_sample_merged = EditDataSample.stack([data_sample1, data_sample2])
        assert (data_sample_merged.img == torch.stack(
            [data_sample1.img, data_sample2.img])).all()
        assert (data_sample_merged.gt_label.label == torch.LongTensor(
            [[1], [2]])).all()
        assert data_sample_merged.mode == ['a', 'b']
        assert data_sample_merged.metainfo == dict(
            channel_order=['rgb', 'rgb'], color_flag=['color', 'color'])
        assert len(data_sample_merged) == 2

        # test split
        data_sample_merged.sample_model = 'ema'
        data_sample_merged.fake_img = EditDataSample(
            img=torch.randn(2, 3, 4, 4))

        data_splited_1, data_splited_2 = data_sample_merged.split(True)
        assert (data_splited_1.gt_label.label == 1).all()
        assert (data_splited_2.gt_label.label == 2).all()
        assert (data_splited_1.img.shape == data_sample1.img.shape)
        assert (data_splited_2.img.shape == data_sample2.img.shape)
        assert (data_splited_1.img == data_sample1.img).all()
        assert (data_splited_2.img == data_sample2.img).all()
        assert (data_splited_1.metainfo == dict(
            channel_order='rgb', color_flag='color'))
        assert (data_splited_2.metainfo == dict(
            channel_order='rgb', color_flag='color'))
        assert data_splited_1.sample_model == 'ema'
        assert data_splited_2.sample_model == 'ema'
        assert data_splited_1.fake_img.img.shape == (3, 4, 4)
        assert data_splited_2.fake_img.img.shape == (3, 4, 4)

        with self.assertRaises(TypeError):
            data_sample_merged.split()

        # test stack and split when batch size is 1
        data_sample = EditDataSample()
        data_sample.set_gt_label(3)
        data_sample.set_tensor_data({'img': torch.randn(3, 4, 5)})
        data_sample.set_data({'mode': 'c'})
        data_sample.set_metainfo({
            'channel_order': 'rgb',
            'color_flag': 'color'
        })

        data_sample_merged = EditDataSample.stack([data_sample])
        assert (data_sample_merged.img == torch.stack([data_sample.img])).all()
        assert (data_sample_merged.gt_label.label == torch.LongTensor(
            [[3]])).all()
        assert data_sample_merged.mode == ['c']
        assert data_sample_merged.metainfo == dict(
            channel_order=['rgb'], color_flag=['color'])
        assert len(data_sample_merged) == 1

        # test split
        data_splited = data_sample_merged.split()
        assert len(data_splited) == 1
        data_splited = data_splited[0]
        assert (data_splited.gt_label.label == 3).all()
        assert (data_splited.img == data_sample.img).all()
        assert (data_splited.metainfo == dict(
            channel_order='rgb', color_flag='color'))

    def test_len(self):
        empty_data = EditDataSample(sample_kwargs={'a': 'a'})
        assert len(empty_data) == 1

        empty_data = EditDataSample()
        assert len(empty_data) == 1

        empty_data = EditDataSample(
            img=torch.randn(3, 3), metainfo=dict(img_shape=[3, 3]))
        assert len(empty_data) == 1
