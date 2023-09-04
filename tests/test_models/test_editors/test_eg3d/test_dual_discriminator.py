# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy
from unittest import TestCase

import pytest
import torch

from mmagic.models.editors.eg3d.dual_discriminator import DualDiscriminator


class TestEG3DDiscriminator(TestCase):

    def setUp(self):
        self.default_cfg = dict(in_size=32, cond_size=25)

    def test_init(self):
        cfg = deepcopy(self.default_cfg)
        disc = DualDiscriminator(**cfg)
        self.assertEqual(disc.convs[0][0].conv.weight.shape[1], 6)
        self.assertTrue(disc.use_dual_disc)

        cfg = deepcopy(self.default_cfg)
        cfg['use_dual_disc'] = False
        cfg['img_channels'] = 2
        disc = DualDiscriminator(**cfg)
        self.assertEqual(disc.convs[0][0].conv.weight.shape[1], 2)
        self.assertFalse(disc.use_dual_disc)

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_forward(self):
        cfg = deepcopy(self.default_cfg)
        disc = DualDiscriminator(**cfg)
        img, img_raw = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 16, 16)
        cond = torch.randn(2, 25)
        out = disc(img, img_raw, cond)
        self.assertEqual(out.shape, (2, 1))
        # test raise error with img_raw is None
        with self.assertRaises(AssertionError):
            disc(img, None, cond)

        # test cond_noise is not None
        cfg = deepcopy(self.default_cfg)
        cfg['disc_c_noise'] = 0.2
        disc = DualDiscriminator(**cfg)
        img, img_raw = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 16, 16)
        cond = torch.randn(2, 25)
        out = disc(img, img_raw, cond)
        self.assertEqual(out.shape, (2, 1))

        # test input_bgr2rgb
        cfg = deepcopy(self.default_cfg)
        cfg['input_bgr2rgb'] = True
        disc = DualDiscriminator(**cfg)
        img, img_raw = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 16, 16)
        cond = torch.randn(2, 25)
        out = disc(img, img_raw, cond)
        self.assertEqual(out.shape, (2, 1))

        # test img_raw is None + use_dual_disc is False
        cfg = deepcopy(self.default_cfg)
        cfg['use_dual_disc'] = False
        disc = DualDiscriminator(**cfg)
        img = torch.randn(2, 3, 32, 32)
        cond = torch.randn(2, 25)
        out = disc(img, None, cond)
        self.assertEqual(out.shape, (2, 1))

        # test img_raw is None + use_dual_disc is False + input_bgr2rgb
        cfg = deepcopy(self.default_cfg)
        cfg['use_dual_disc'] = False
        cfg['input_bgr2rgb'] = True
        disc = DualDiscriminator(**cfg)
        img = torch.randn(2, 3, 32, 32)
        cond = torch.randn(2, 25)
        out = disc(img, None, cond)
        self.assertEqual(out.shape, (2, 1))

        # test cond is None
        cfg = deepcopy(self.default_cfg)
        cfg['cond_size'] = None
        disc = DualDiscriminator(**cfg)
        out = disc(img, img_raw, None)
        self.assertEqual(out.shape, (2, 1))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
