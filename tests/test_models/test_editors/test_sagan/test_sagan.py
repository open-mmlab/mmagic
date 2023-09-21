# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch.optim import SGD

from mmagic.models import SAGAN, DataPreprocessor
from mmagic.registry import MODELS
from mmagic.structures import DataSample

generator = dict(
    type='SAGANGenerator',
    # noise_size=10,
    num_classes=10,
    output_scale=32,
    base_channels=256,
    attention_cfg=dict(type='SelfAttentionBlock'),
    attention_after_nth_block=2,
    with_spectral_norm=True)
discriminator = dict(
    type='ProjDiscriminator',
    num_classes=10,
    input_scale=32,
    base_channels=128,
    attention_cfg=dict(type='SelfAttentionBlock'),
    attention_after_nth_block=1,
    with_spectral_norm=True)


class TestSAGAN(TestCase):

    def test_init(self):
        gan = SAGAN(
            noise_size=10,
            num_classes=10,
            data_preprocessor=DataPreprocessor(),
            generator=generator,
            discriminator=discriminator,
            generator_steps=1,
            discriminator_steps=4)

        self.assertIsInstance(gan, SAGAN)
        self.assertIsInstance(gan.data_preprocessor, DataPreprocessor)

        # test only generator have noise size
        gen_cfg = deepcopy(generator)
        gen_cfg['noise_size'] = 10
        gan = SAGAN(
            noise_size=10,
            generator=gen_cfg,
            discriminator=discriminator,
            data_preprocessor=DataPreprocessor())
        self.assertEqual(gan.noise_size, 10)

        # test init with nn.Module
        gen_cfg = deepcopy(generator)
        gen_cfg['noise_size'] = 10
        disc_cfg = deepcopy(discriminator)
        gen = MODELS.build(gen_cfg)
        disc = MODELS.build(disc_cfg)
        gan = SAGAN(
            noise_size=10,
            generator=gen,
            discriminator=disc,
            data_preprocessor=DataPreprocessor())
        self.assertEqual(gan.generator, gen)
        self.assertEqual(gan.discriminator, disc)

        # test init without discriminator
        gan = SAGAN(
            noise_size=10, generator=gen, data_preprocessor=DataPreprocessor())
        self.assertEqual(gan.discriminator, None)

        # test init with different num_classes
        gan = SAGAN(
            num_classes=10,
            data_preprocessor=DataPreprocessor(),
            generator=generator,
            discriminator=discriminator,
            generator_steps=1,
            discriminator_steps=4)

    def test_train_step(self):
        # prepare model
        accu_iter = 1
        n_disc = 1
        message_hub = MessageHub.get_instance('mmgen')
        gan = SAGAN(
            noise_size=10,
            generator=generator,
            discriminator=discriminator,
            data_preprocessor=DataPreprocessor(),
            discriminator_steps=n_disc)
        # prepare messageHub
        message_hub.update_info('iter', 0)
        # prepare optimizer
        gen_optim = SGD(gan.generator.parameters(), lr=0.1)
        disc_optim = SGD(gan.discriminator.parameters(), lr=0.1)
        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(gen_optim, accumulative_counts=accu_iter),
            discriminator=OptimWrapper(
                disc_optim, accumulative_counts=accu_iter))
        # prepare inputs
        img = torch.randn(3, 16, 16)
        label = torch.randint(0, 10, (1, ))

        data_sample = DataSample(gt_img=img)
        data_sample.set_gt_label(label)

        data = dict(inputs=dict(), data_samples=[data_sample])

        # simulate train_loop here
        for idx in range(n_disc * accu_iter):
            message_hub.update_info('iter', idx)
            log = gan.train_step(data, optim_wrapper_dict)
            if (idx + 1) == n_disc * accu_iter:
                # should update at after (n_disc * accu_iter)
                self.assertEqual(
                    set(log.keys()),
                    set([
                        'loss', 'loss_disc_fake', 'loss_disc_real', 'loss_gen'
                    ]))
            else:
                # should not update when discriminator's updating is unfinished
                self.assertEqual(
                    log.keys(),
                    set(['loss', 'loss', 'loss_disc_fake', 'loss_disc_real']))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
