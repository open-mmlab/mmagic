# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest import TestCase

import pytest
import torch
from mmengine import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict

from mmedit.models import GCFSRGAN, GenDataPreprocessor


class TestGCFSRGAN(TestCase):

    @classmethod
    def setup_class(cls):
        cls.generator_cfg = dict(
            type='GCFSR', out_size=128, num_style_feat=128)
        cls.disc_cfg = dict(type='StyleGAN2Discriminator', in_size=128)

        # reg params
        ema_half_life = 10.  # G_smoothing_kimg

        cls.ema_config = dict(
            type='ExponentialMovingAverage',
            interval=1,
            momentum=0.5**(32. / (ema_half_life * 1000.)))

        cls.pixel_loss_cfg = dict(
            type='L1Loss', loss_weight=1.0, reduction='mean')
        cls.perceptual_loss_cfg = dict(
            type='PerceptualLoss',
            layer_weights={'21': 1.0},
            vgg_type='vgg16',
            perceptual_weight=1e-2,
            style_weight=0,
            norm_img=True,
            criterion='l1',
            pretrained='torchvision://vgg16')
        cls.gan_loss_cfg = dict(
            type='GANLoss', gan_type='wgan_softplus', loss_weight=1e-2)

        cls.rescale_list_cfg = [64, 64, 64, 64, 32, 32, 16, 16, 8, 4]

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_gcfsargan_cpu(self):
        accu_iter = 1
        message_hub = MessageHub.get_instance('test-s2')
        gcfsr = GCFSRGAN(
            self.generator_cfg,
            self.disc_cfg,
            data_preprocessor=GenDataPreprocessor(),
            ema_config=self.ema_config,
            pixel_loss=self.pixel_loss_cfg,
            gan_loss=self.gan_loss_cfg,
            perceptual_loss=self.perceptual_loss_cfg,
            rescale_list=self.rescale_list_cfg)

        optimizer_g = torch.optim.SGD(gcfsr.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            gcfsr.discriminator.parameters(), lr=0.01)

        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(optimizer_g, accumulative_counts=accu_iter),
            discriminator=OptimWrapper(
                optimizer_d, accumulative_counts=accu_iter))

        # prepare inputs
        img = torch.randn(1, 3, 128, 128)
        data = dict(inputs=dict(img=img))

        # simulate train_loop here
        message_hub.update_info('iter', 0)
        _ = gcfsr.train_step(data, optim_wrapper_dict)
