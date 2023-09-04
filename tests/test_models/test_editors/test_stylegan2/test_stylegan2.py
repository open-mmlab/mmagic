# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest import TestCase

import pytest
import torch
from mmengine import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict

from mmagic.models import DataPreprocessor, StyleGAN2
from mmagic.structures import DataSample


class TestStyleGAN2(TestCase):

    @classmethod
    def setup_class(cls):
        cls.generator_cfg = dict(
            type='StyleGANv2Generator', out_size=32, style_channels=16)
        cls.disc_cfg = dict(type='StyleGAN2Discriminator', in_size=32)

        # reg params
        d_reg_interval = 16
        g_reg_interval = 4
        ema_half_life = 10.  # G_smoothing_kimg

        cls.ema_config = dict(
            type='ExponentialMovingAverage',
            interval=1,
            momentum=1. - (0.5**(32. / (ema_half_life * 1000.))))

        cls.loss_config = dict(
            r1_loss_weight=10. / 2. * d_reg_interval,
            r1_interval=d_reg_interval,
            norm_mode='HWC',
            g_reg_interval=g_reg_interval,
            g_reg_weight=2. * g_reg_interval,
            pl_batch_shrink=2)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_stylegan2_cpu(self):
        accu_iter = 1
        message_hub = MessageHub.get_instance('test-s2')
        stylegan2 = StyleGAN2(
            self.generator_cfg,
            self.disc_cfg,
            data_preprocessor=DataPreprocessor(),
            ema_config=self.ema_config,
            loss_config=self.loss_config)

        optimizer_g = torch.optim.SGD(
            stylegan2.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            stylegan2.discriminator.parameters(), lr=0.01)

        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(optimizer_g, accumulative_counts=accu_iter),
            discriminator=OptimWrapper(
                optimizer_d, accumulative_counts=accu_iter))

        # prepare inputs
        img = torch.randn(3, 32, 32)
        data = dict(inputs=dict(), data_samples=[DataSample(gt_img=img)])

        # simulate train_loop here
        message_hub.update_info('iter', 0)
        _ = stylegan2.train_step(data, optim_wrapper_dict)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
