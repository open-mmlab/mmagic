# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy
from unittest import TestCase

import pytest
import torch
from mmengine import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch.optim import Adam

from mmagic.models import DataPreprocessor, DeblurGanV2
from mmagic.models.losses import PerceptualLoss
from mmagic.models.losses.adv_loss import DiscLossWGANGP
from mmagic.registry import MODELS
from mmagic.structures import DataSample

generator = dict(
    type='DeblurGanV2Generator',
    backbone='FPNMobileNet',
    norm_layer='instance',
    output_ch=3,
    num_filter=64,
    num_filter_fpn=128,
)
discriminator = dict(
    type='DeblurGanV2Discriminator',
    backbone='DoubleGan',
    norm_layer='instance',
    d_layers=3,
)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
class TestDeblurGanV2(TestCase):

    def test_init(self):
        gan = DeblurGanV2(
            generator=generator,
            discriminator=discriminator,
            pixel_loss=dict(
                type='PerceptualLoss',
                layer_weights={'14': 1},
                criterion='mse'),
            adv_lambda=0.001,
            warmup_num=3,
            disc_loss=dict(type='AdvLoss', loss_type='wgan-gp'),
            data_preprocessor=DataPreprocessor())

        self.assertIsInstance(gan, DeblurGanV2)
        self.assertIsInstance(gan.data_preprocessor, DataPreprocessor)
        self.assertIsInstance(gan.pixel_loss, PerceptualLoss)
        self.assertIsInstance(gan.disc_loss, DiscLossWGANGP)
        gen_cfg = deepcopy(generator)
        disc_cfg = deepcopy(discriminator)
        gen = MODELS.build(gen_cfg)
        disc = MODELS.build(disc_cfg)
        gan = DeblurGanV2(
            generator=gen,
            discriminator=disc,
            pixel_loss=dict(
                type='PerceptualLoss',
                layer_weights={'14': 1},
                criterion='mse'),
            adv_lambda=0.001,
            warmup_num=3,
            disc_loss=dict(type='AdvLoss', loss_type='wgan-gp'),
            data_preprocessor=DataPreprocessor())
        self.assertEqual(gan.generator, gen)
        self.assertEqual(gan.discriminator, disc)

        # test init without discriminator
        gan = DeblurGanV2(
            generator=gen_cfg, data_preprocessor=DataPreprocessor())
        self.assertEqual(gan.discriminator, None)

    def test_train_step(self):
        # prepare model
        accu_iter = 1
        n_disc = 1
        message_hub = MessageHub.get_instance('test-lsgan')
        gan = DeblurGanV2(
            generator=generator,
            discriminator=discriminator,
            pixel_loss=dict(
                type='PerceptualLoss',
                layer_weights={'14': 1},
                criterion='mse'),
            adv_lambda=0.001,
            warmup_num=3,
            disc_loss=dict(type='AdvLoss', loss_type='wgan-gp'),
            data_preprocessor=DataPreprocessor())
        # prepare messageHub
        message_hub.update_info('iter', 0)
        # prepare optimizer
        gen_optim = Adam(
            gan.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        disc_optim = Adam(
            list(gan.discriminator.patch_gan.parameters()) +
            list(gan.discriminator.full_gan.parameters()),
            lr=0.0001,
            betas=(0.5, 0.999))
        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(gen_optim, accumulative_counts=accu_iter),
            discriminator=OptimWrapper(
                disc_optim, accumulative_counts=accu_iter))
        # prepare inputs
        img = torch.randn(3, 256, 256)
        data = dict(inputs=[img], data_samples=[DataSample(gt_img=img)])

        # simulate train_loop here
        for idx in range(n_disc * accu_iter):
            message_hub.update_info('iter', idx)
            log = gan.train_step(data, optim_wrapper_dict)
            if (idx + 1) == n_disc * accu_iter:
                # should update at after (n_disc * accu_iter)
                self.assertEqual(
                    set(log.keys()),
                    set(['loss_d', 'loss_g_content', 'loss_g_adv', 'loss_g']))
            else:
                # should not update when discriminator's updating is unfinished
                self.assertEqual(
                    log.keys(), set(['loss_g_content', 'loss_g_adv',
                                     'loss_g']))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
