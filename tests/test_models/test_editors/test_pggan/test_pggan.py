# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine import MessageHub

from mmagic.engine import PGGANOptimWrapperConstructor
from mmagic.models import ProgressiveGrowingGAN
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


class TestPGGAN(TestCase):

    generator_cfg = dict(
        type='PGGANGenerator',
        noise_size=8,
        out_scale=16,
        base_channels=32,
        max_channels=32)

    discriminator_cfg = dict(
        type='PGGANDiscriminator', in_scale=16, label_size=0)

    data_preprocessor = dict(type='DataPreprocessor')

    nkimgs_per_scale = {'4': 0.004, '8': 0.008, '16': 0.016}

    lr_schedule = dict(generator={'8': 0.0015}, discriminator={'8': 0.0015})
    optim_wrapper_cfg = dict(
        generator=dict(
            optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
        discriminator=dict(
            optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
        lr_schedule=lr_schedule)

    def test_pggan_cpu(self):
        message_hub = MessageHub.get_instance('test-pggan')
        message_hub.update_info('iter', 0)

        # test default config
        pggan = ProgressiveGrowingGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            data_preprocessor=self.data_preprocessor,
            nkimgs_per_scale=self.nkimgs_per_scale,
            ema_config=dict(interval=1))

        constructor = PGGANOptimWrapperConstructor(self.optim_wrapper_cfg)
        optim_wrapper_dict = constructor(pggan)

        data_batch = dict(
            inputs=dict(),
            data_samples=[
                DataSample(gt_img=torch.randn(3, 16, 16)) for _ in range(3)
            ])

        for iter_num in range(6):
            pggan.train_step(data_batch, optim_wrapper_dict)
            # print(iter_num, pggan._next_scale_int)
            if iter_num in [0, 1]:
                assert pggan.curr_scale[0] == 4
            elif iter_num in [2, 3]:
                assert pggan.curr_scale[0] == 8
            elif iter_num in [4, 5]:
                assert pggan.curr_scale[0] == 16

            if iter_num == 2:
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 3:
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 5:
                assert np.isclose(pggan._actual_nkimgs[-1], 0.012, atol=1e-8)

        # test forward
        outputs = pggan.forward(dict(img=torch.randn(3, 3, 16, 16)))
        assert len(outputs) == 3
        assert all(['gt_img' in out for out in outputs])

        outputs = pggan.forward(dict(num_batches=2))
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 16, 16) for out in outputs])

        outputs = pggan.forward(
            dict(
                num_batches=2,
                return_noise=True,
                transition_weight=0.2,
                sample_model='ema'))
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 16, 16) for out in outputs])

        outputs = pggan.forward(dict(num_batches=2, sample_model='orig'))
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 16, 16) for out in outputs])

        outputs = pggan.forward(dict(num_batches=2, sample_model='ema/orig'))
        assert len(outputs) == 2
        assert all([out.ema.fake_img.shape == (3, 16, 16) for out in outputs])
        assert all([out.orig.fake_img.shape == (3, 16, 16) for out in outputs])

        outputs = pggan.forward(dict(num_batches=2, curr_scale=8))
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 8, 8) for out in outputs])

        outputs = pggan.forward(dict(noise=torch.randn(2, 8)))
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 16, 16) for out in outputs])

        outputs = pggan.forward(torch.randn(2, 8))
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 16, 16) for out in outputs])

        # test train_step with error
        with pytest.raises(RuntimeError):
            data_batch = dict(
                inputs=dict(),
                data_samples=[
                    DataSample(gt_img=torch.randn(3, 4, 32)) for _ in range(3)
                ])
            _ = pggan.train_step(data_batch, optim_wrapper_dict)

        # test train_step without ema
        pggan = ProgressiveGrowingGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            data_preprocessor=self.data_preprocessor,
            nkimgs_per_scale=self.nkimgs_per_scale)
        optim_wrapper_dict = constructor(pggan)
        data_batch = dict(
            inputs=dict(),
            data_samples=[
                DataSample(gt_img=torch.randn(3, 16, 16)) for _ in range(3)
            ])
        pggan.train_step(data_batch, optim_wrapper_dict)

        # test train_step with disc_step != 1
        pggan._disc_steps = 2
        pggan.train_step(data_batch, optim_wrapper_dict)

        # test default configs
        pggan = ProgressiveGrowingGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            data_preprocessor=self.data_preprocessor,
            nkimgs_per_scale=self.nkimgs_per_scale,
            interp_real=dict(mode='bicubic'),
            ema_config=dict(interval=1))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pggan_cuda(self):
        pggan = ProgressiveGrowingGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            data_preprocessor=self.data_preprocessor,
            nkimgs_per_scale=self.nkimgs_per_scale,
            ema_config=dict(interval=1)).cuda()

        constructor = PGGANOptimWrapperConstructor(self.optim_wrapper_cfg)
        optim_wrapper_dict = constructor(pggan)

        data_batch = dict(
            inputs=dict(),
            data_samples=[
                DataSample(gt_img=torch.randn(3, 16, 16)) for _ in range(3)
            ])

        for iter_num in range(6):
            pggan.train_step(data_batch, optim_wrapper_dict)
            if iter_num in [0, 1]:
                assert pggan.curr_scale[0] == 4
            elif iter_num in [2, 3]:
                assert pggan.curr_scale[0] == 8
            elif iter_num in [4, 5]:
                assert pggan.curr_scale[0] == 16

            if iter_num == 2:
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 3:
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 5:
                assert np.isclose(pggan._actual_nkimgs[-1], 0.012, atol=1e-8)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
