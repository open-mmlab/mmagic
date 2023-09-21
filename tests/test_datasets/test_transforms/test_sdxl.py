# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

import numpy as np
import torch
from PIL import Image

from mmagic.registry import TRANSFORMS


class TestComputeTimeIds(TestCase):

    def test_register(self):
        self.assertIn('ComputeTimeIds', TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/sd/color.jpg')
        img = Image.open(img_path)
        data = {
            'img': np.array(img),
            'ori_img_shape': [32, 32],
            'img_crop_bbox': [0, 0, 32, 32]
        }

        # test transform
        trans = TRANSFORMS.build(dict(type='ComputeTimeIds'))
        data = trans(data)
        self.assertIsInstance(data['time_ids'], torch.Tensor)
        self.assertListEqual(
            list(data['time_ids'].numpy()),
            [32, 32, 0, 0, img.height, img.width])

        assert trans.__repr__() == (trans.__class__.__name__ + '(key=img)')


class TestRandomCropXL(TestCase):
    crop_size = 32

    def test_register(self):
        self.assertIn('RandomCropXL', TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/sd/color.jpg')
        data = {'img': np.array(Image.open(img_path))}

        # test transform
        trans = TRANSFORMS.build(
            dict(type='RandomCropXL', size=self.crop_size))
        data = trans(data)
        self.assertIn('img_crop_bbox', data)
        assert len(data['img_crop_bbox']) == 4
        assert data['img'].shape[0] == data['img'].shape[1] == self.crop_size
        upper, left, lower, right = data['img_crop_bbox']
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data['img']),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))

        assert trans.__repr__() == (
            trans.__class__.__name__ +
            f'(size={(self.crop_size, self.crop_size)},'
            f" keys=['img'])")

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/sd/color.jpg')
        data = {
            'img': np.array(Image.open(img_path)),
            'condition_img': np.array(Image.open(img_path))
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type='RandomCropXL',
                size=self.crop_size,
                keys=['img', 'condition_img']))
        data = trans(data)
        self.assertIn('img_crop_bbox', data)
        assert len(data['img_crop_bbox']) == 4
        assert data['img'].shape[0] == data['img'].shape[1] == self.crop_size
        upper, left, lower, right = data['img_crop_bbox']
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data['img']),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))
        np.equal(np.array(data['img']), np.array(data['condition_img']))


class TestFlipXL(TestCase):

    def test_register(self):
        self.assertIn('FlipXL', TRANSFORMS)

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/sd/color.jpg')
        data = {
            'img': np.array(Image.open(img_path)),
            'img_crop_bbox': [0, 0, 10, 10]
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(type='FlipXL', flip_ratio=1., keys=['img']))
        data = trans(data)
        self.assertIn('img_crop_bbox', data)
        assert len(data['img_crop_bbox']) == 4
        self.assertListEqual(
            data['img_crop_bbox'],
            [0, data['img'].shape[1] - 10, 10, data['img'].shape[1] - 0])

        np.equal(
            np.array(data['img']),
            np.array(Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)))

        assert trans.__repr__() == (
            trans.__class__.__name__ +
            "(keys=['img'], flip_ratio=1.0, direction=horizontal)")

        # test transform p=0.0
        data = {
            'img': np.array(Image.open(img_path)),
            'img_crop_bbox': [0, 0, 10, 10]
        }
        trans = TRANSFORMS.build(
            dict(type='FlipXL', flip_ratio=0., keys='img'))
        data = trans(data)
        self.assertIn('img_crop_bbox', data)
        self.assertListEqual(data['img_crop_bbox'], [0, 0, 10, 10])

        np.equal(np.array(data['img']), np.array(Image.open(img_path)))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/sd/color.jpg')
        data = {
            'img': np.array(Image.open(img_path)),
            'condition_img': np.array(Image.open(img_path)),
            'img_crop_bbox': [0, 0, 10, 10],
            'condition_img_crop_bbox': [0, 0, 10, 10]
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(type='FlipXL', flip_ratio=1., keys=['img', 'condition_img']))
        data = trans(data)
        self.assertIn('img_crop_bbox', data)
        assert len(data['img_crop_bbox']) == 4
        self.assertListEqual(
            data['img_crop_bbox'],
            [0, data['img'].shape[1] - 10, 10, data['img'].shape[1] - 0])

        np.equal(
            np.array(data['img']),
            np.array(Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)))
        np.equal(np.array(data['img']), np.array(data['condition_img']))


class TestResizeEdge(TestCase):

    def test_transform(self):
        results = dict(img=np.random.randint(0, 256, (128, 256, 3), np.uint8))

        # test resize short edge by default.
        cfg = dict(type='ResizeEdge', scale=224)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 448, 3))

        # test resize long edge.
        cfg = dict(type='ResizeEdge', scale=224, edge='long')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (112, 224, 3))

        # test resize width.
        cfg = dict(type='ResizeEdge', scale=224, edge='width')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (112, 224, 3))

        # test resize height.
        cfg = dict(type='ResizeEdge', scale=224, edge='height')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 448, 3))

        # test invalid edge
        with self.assertRaisesRegex(AssertionError, 'Invalid edge "hi"'):
            cfg = dict(type='ResizeEdge', scale=224, edge='hi')
            TRANSFORMS.build(cfg)

    def test_repr(self):
        cfg = dict(type='ResizeEdge', scale=224, edge='height')
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'ResizeEdge(scale=224, edge=height, backend=cv2, '
            'interpolation=bilinear)')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
