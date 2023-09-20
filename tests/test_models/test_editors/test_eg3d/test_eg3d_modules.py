# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch

from mmagic.models.editors.eg3d.eg3d_modules import (SuperResolutionModule,
                                                     TriPlaneBackbone)


class TestTriPlaneBackbone(TestCase):

    def setUp(self):
        self.default_cfg = dict(
            out_size=32,
            noise_size=10,
            out_channels=9,
            style_channels=16,
            num_mlps=2,
            cond_size=4,
            cond_scale=1,
            cond_mapping_channels=4,
            zero_cond_input=False)

    def test_init(self):
        cfg_ = deepcopy(self.default_cfg)
        backbone = TriPlaneBackbone(**cfg_)
        self.assertEqual(backbone.cond_scale, 1)
        self.assertEqual(backbone.zero_cond_input, False)

    def test_mapping(self):
        cfg_ = deepcopy(self.default_cfg)
        backbone = TriPlaneBackbone(**cfg_)
        noise = torch.randn(2, 10)
        label = torch.randn(2, 4)
        style_code = backbone.mapping(noise, label)
        self.assertEqual(style_code.shape, (2, 16))

        # test truncation < 1
        style_code = backbone.mapping(noise, label, truncation=0.5)
        self.assertEqual(style_code.shape, (2, 16))
        style_code = backbone.mapping(noise, label, truncation=0.5)
        self.assertEqual(style_code.shape, (2, 16))

        # test zero_cond_input + label is None
        cfg_ = deepcopy(self.default_cfg)
        cfg_['zero_cond_input'] = True
        backbone = TriPlaneBackbone(**cfg_)
        style_code = backbone.mapping(noise)
        self.assertEqual(style_code.shape, (2, 16))

        # test zero_cond_input + label is not None
        style_code = backbone.mapping(noise, label)
        self.assertEqual(style_code.shape, (2, 16))

        # test cond_size < 0
        cfg_ = deepcopy(self.default_cfg)
        cfg_['cond_size'] = None
        backbone = TriPlaneBackbone(**cfg_)
        # noise = torch.randn(2, 10)
        style_code = backbone.mapping(noise)
        self.assertEqual(style_code.shape, (2, 16))

    def test_synthesis(self):
        cfg_ = deepcopy(self.default_cfg)
        backbone = TriPlaneBackbone(**cfg_)
        noise = torch.randn(2, 10)
        label = torch.randn(2, 4)
        style_code = backbone.mapping(noise, label)
        outputs_forward = backbone(noise, label)
        outputs_synthesis = backbone.synthesis(style_code)
        self.assertTrue((outputs_forward == outputs_synthesis).all())


class TestSuperResolutionModule(TestCase):

    def setUp(self):
        self.default_cfg = dict(
            in_channels=8,
            in_size=4,
            hidden_size=8,
            out_size=8,
            style_channels=5,
            sr_antialias=True)

    def test_init(self):
        cfg_ = deepcopy(self.default_cfg)
        sr_model = SuperResolutionModule(**cfg_)
        self.assertTrue(sr_model.block0.upsample)
        self.assertFalse(sr_model.block1.upsample)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['hidden_size'] = 4
        sr_model = SuperResolutionModule(**cfg_)
        self.assertFalse(sr_model.block0.upsample)
        self.assertTrue(sr_model.block1.upsample)

    def test_forward(self):
        cfg_ = deepcopy(self.default_cfg)
        sr_model = SuperResolutionModule(**cfg_)
        styles = torch.randn(2, 5)
        imgs = torch.randn(2, 3, 4, 4)
        features = torch.randn(2, 8, 4, 4)
        sr_imgs = sr_model(imgs, features, styles)
        self.assertEqual(sr_imgs.shape, (2, 3, 8, 8))

        # test style is a list
        styles = [torch.randn(2, 5)]
        imgs = torch.randn(2, 3, 4, 4)
        features = torch.randn(2, 8, 4, 4)
        sr_imgs = sr_model(imgs, features, styles)
        self.assertEqual(sr_imgs.shape, (2, 3, 8, 8))

        # test style's ndim is 3
        styles = torch.randn(3, 2, 5)
        imgs = torch.randn(2, 3, 4, 4)
        features = torch.randn(2, 8, 4, 4)
        sr_imgs = sr_model(imgs, features, styles)
        self.assertEqual(sr_imgs.shape, (2, 3, 8, 8))

        # test interpolation
        imgs = torch.randn(2, 3, 3, 3)
        features = torch.randn(2, 8, 3, 3)
        sr_imgs = sr_model(imgs, features, styles)
        self.assertEqual(sr_imgs.shape, (2, 3, 8, 8))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
