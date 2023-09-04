# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy
from unittest import TestCase

import pytest
import torch
from mmengine import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict

from mmagic.models import StyleGAN3
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


class TestStyleGAN3(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cfg = dict(
            data_preprocessor=dict(type='DataPreprocessor'),
            generator=dict(
                type='StyleGANv3Generator',
                noise_size=6,
                style_channels=8,
                out_size=16,
                img_channels=3,
                synthesis_cfg=dict(
                    type='SynthesisNetwork',
                    channel_base=1024,
                    channel_max=16,
                    magnitude_ema_beta=0.999)),
            discriminator=dict(
                type='StyleGAN2Discriminator',
                in_size=16,
                channel_multiplier=1),
            ema_config=dict(
                type='RampUpEMA',
                interval=1,
                ema_kimg=10,
                ema_rampup=0.05,
                batch_size=32,
                eps=1e-08,
                start_iter=0),
            loss_config=dict(
                r1_loss_weight=16.0,
                r1_interval=16,
                norm_mode='HWC',
                g_reg_interval=4,
                g_reg_weight=8.0,
                pl_batch_shrink=2))

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_val_and_test_step(self):
        cfg = deepcopy(self.default_cfg)
        stylegan = StyleGAN3(**cfg)

        data = dict(inputs=dict(num_batches=2))
        outputs = stylegan.test_step(data)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].fake_img.data.shape, (3, 16, 16))

        data = dict(inputs=dict(num_batches=2))
        outputs = stylegan.val_step(data)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].fake_img.data.shape, (3, 16, 16))

        eq_cfg = dict(
            compute_eqt_int=True, compute_eqt_frac=True, compute_eqr=True)
        data = dict(inputs=dict(num_batches=2, eq_cfg=eq_cfg, mode='orig'))
        outputs = stylegan.test_step(data)
        outputs = stylegan.val_step(data)

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_train_step(self):
        message_hub = MessageHub.get_instance('test-s3-train-step')
        cfg = deepcopy(self.default_cfg)
        stylegan = StyleGAN3(**cfg)
        optimizer_g = torch.optim.SGD(stylegan.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            stylegan.discriminator.parameters(), lr=0.01)
        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(optimizer_g, accumulative_counts=1),
            discriminator=OptimWrapper(optimizer_d, accumulative_counts=1))

        img = torch.randn(3, 16, 16)
        data = dict(inputs=dict(), data_samples=[DataSample(gt_img=img)])
        message_hub.update_info('iter', 0)
        _ = stylegan.train_step(data, optim_wrapper_dict)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
