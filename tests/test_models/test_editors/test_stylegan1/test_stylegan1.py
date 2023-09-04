# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine import MessageHub

from mmagic.engine import PGGANOptimWrapperConstructor
from mmagic.models import StyleGAN1
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


class TestStyleGAN1(TestCase):
    style_channels = 16
    generator_cfg = dict(
        type='StyleGAN1Generator', out_size=64, style_channels=16)
    discriminator_cfg = dict(type='StyleGAN1Discriminator', in_size=64)

    nkimgs_per_scale = {'16': 0.004, '32': 0.008, '64': 0.016}

    data_preprocessor = dict(type='DataPreprocessor')

    lr_schedule = dict(generator={'32': 0.0015}, discriminator={'32': 0.0015})
    optim_wrapper_cfg = dict(
        generator=dict(
            optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
        discriminator=dict(
            optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
        lr_schedule=lr_schedule)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_stylegan_cpu(self):
        message_hub = MessageHub.get_instance('test-s1')
        message_hub.update_info('iter', 0)

        # test default config
        stylegan = StyleGAN1(
            self.generator_cfg,
            self.discriminator_cfg,
            style_channels=self.style_channels,
            data_preprocessor=self.data_preprocessor,
            nkimgs_per_scale=self.nkimgs_per_scale,
            ema_config=dict(interval=1))

        constructor = PGGANOptimWrapperConstructor(self.optim_wrapper_cfg)
        optim_wrapper_dict = constructor(stylegan)

        data_batch = dict(
            inputs=dict(),
            data_samples=[
                DataSample(gt_img=torch.randn(3, 64, 64)) for _ in range(3)
            ])

        for iter_num in range(6):
            stylegan.train_step(data_batch, optim_wrapper_dict)
            if iter_num in [0, 1]:
                assert stylegan.curr_scale[0] == 16
            elif iter_num in [2, 3]:
                assert stylegan.curr_scale[0] == 32
            elif iter_num in [4, 5]:
                assert stylegan.curr_scale[0] == 64

            if iter_num == 2:
                assert np.isclose(stylegan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 3:
                assert np.isclose(stylegan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 5:
                assert np.isclose(
                    stylegan._actual_nkimgs[-1], 0.012, atol=1e-8)

        # test forward
        outputs = stylegan.forward(dict(num_batches=2))
        # assert outputs.shape == (2, 3, 64, 64)
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 64, 64) for out in outputs])

        outputs = stylegan.forward(
            dict(
                num_batches=2,
                return_noise=True,
                transition_weight=0.2,
                sample_model='ema'))
        # assert outputs.shape == (2, 3, 64, 64)
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 64, 64) for out in outputs])

        outputs = stylegan.forward(dict(num_batches=2, sample_model='orig'))
        # assert outputs.shape == (2, 3, 64, 64)
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 64, 64) for out in outputs])

        outputs = stylegan.forward(
            dict(num_batches=2, sample_model='ema/orig'))
        # assert isinstance(outputs, dict)
        # assert list(outputs.keys()) == ['ema', 'orig']
        # assert all([o.shape == (2, 3, 64, 64) for o in outputs.values()])
        assert len(outputs) == 2
        assert all([out.ema.fake_img.shape == (3, 64, 64) for out in outputs])
        assert all([out.orig.fake_img.shape == (3, 64, 64) for out in outputs])

        outputs = stylegan.forward(dict(num_batches=2, curr_scale=8))
        # assert outputs.shape == (2, 3, 8, 8)
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 8, 8) for out in outputs])

        outputs = stylegan.forward(dict(noise=torch.randn(2, 16)))
        # assert outputs.shape == (2, 3, 64, 64)
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 64, 64) for out in outputs])

        outputs = stylegan.forward(torch.randn(2, 16))
        # assert outputs.shape == (2, 3, 64, 64)
        assert len(outputs) == 2
        assert all([out.fake_img.shape == (3, 64, 64) for out in outputs])

        # test train_step with error
        with pytest.raises(RuntimeError):
            data_batch = dict(
                inputs=dict(),
                data_samples=[
                    DataSample(gt_img=torch.randn(3, 4, 32)) for _ in range(3)
                ])
            _ = stylegan.train_step(data_batch, optim_wrapper_dict)

        # test train_step without ema
        stylegan = StyleGAN1(
            self.generator_cfg,
            self.discriminator_cfg,
            style_channels=self.style_channels,
            data_preprocessor=self.data_preprocessor,
            nkimgs_per_scale=self.nkimgs_per_scale)
        optim_wrapper_dict = constructor(stylegan)

        data_batch = dict(
            inputs=dict(),
            data_samples=[
                DataSample(gt_img=torch.randn(3, 64, 64)) for _ in range(3)
            ])
        stylegan.train_step(data_batch, optim_wrapper_dict)

        # test train_step with disc_step != 1
        stylegan._disc_steps = 2
        stylegan.train_step(data_batch, optim_wrapper_dict)

        # test default configs
        stylegan = StyleGAN1(
            self.generator_cfg,
            self.discriminator_cfg,
            style_channels=self.style_channels,
            data_preprocessor=self.data_preprocessor,
            nkimgs_per_scale=self.nkimgs_per_scale,
            interp_real=dict(mode='bicubic'),
            ema_config=dict(interval=1))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_stylegan_cuda(self):
        stylegan = StyleGAN1(
            self.generator_cfg,
            self.discriminator_cfg,
            style_channels=self.style_channels,
            data_preprocessor=self.data_preprocessor,
            nkimgs_per_scale=self.nkimgs_per_scale,
            ema_config=dict(interval=1)).cuda()

        constructor = PGGANOptimWrapperConstructor(self.optim_wrapper_cfg)
        optim_wrapper_dict = constructor(stylegan)

        data_batch = dict(
            inputs=dict(),
            data_samples=[
                DataSample(gt_img=torch.randn(3, 64, 64)) for _ in range(3)
            ])

        for iter_num in range(6):
            stylegan.train_step(data_batch, optim_wrapper_dict)
            if iter_num in [0, 1]:
                assert stylegan.curr_scale[0] == 16
            elif iter_num in [2, 3]:
                assert stylegan.curr_scale[0] == 32
            elif iter_num in [4, 5]:
                assert stylegan.curr_scale[0] == 64

            if iter_num == 2:
                assert np.isclose(stylegan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 3:
                assert np.isclose(stylegan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 5:
                assert np.isclose(
                    stylegan._actual_nkimgs[-1], 0.012, atol=1e-8)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
