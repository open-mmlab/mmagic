# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

from mmagic.engine import SinGANOptimWrapperConstructor
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()


class TestSinGANOptimWrapperConstructor(TestCase):

    singan_cfg = dict(
        type='SinGAN',
        num_scales=2,
        data_preprocessor=dict(
            type='DataPreprocessor', non_image_keys=['input_sample']),
        generator=dict(
            type='SinGANMultiScaleGenerator',
            in_channels=3,
            out_channels=3,
            num_scales=2),
        discriminator=dict(
            type='SinGANMultiScaleDiscriminator', in_channels=3, num_scales=2),
        noise_weight_init=0.1,
        iters_per_scale=20,
        test_pkl_data=None)

    optim_wrapper_cfg = dict(
        generator=dict(
            optimizer=dict(type='Adam', lr=0.0005, betas=(0.5, 0.999))),
        discriminator=dict(
            optimizer=dict(type='Adam', lr=0.0005, betas=(0.5, 0.999))))

    def test(self):
        singan = MODELS.build(self.singan_cfg)
        optim_wrapper_dict_builder = SinGANOptimWrapperConstructor(
            self.optim_wrapper_cfg)
        optim_wrapper_dict = optim_wrapper_dict_builder(singan)
        optim_keys = set(optim_wrapper_dict.keys())
        self.assertEqual(
            optim_keys,
            set([
                f'{model}_{scale}' for model in ['generator', 'discriminator']
                for scale in range(2 + 1)
            ]))

        # test singan is Wrapper
        singan_with_wrapper = MagicMock(
            module=singan,
            generator=MagicMock(module=singan.generator),
            discriminator=MagicMock(module=singan.discriminator))
        optim_wrapper_dict = optim_wrapper_dict_builder(singan_with_wrapper)

        # test raise error
        with self.assertRaises(TypeError):
            optim_wrapper_dict_builder = SinGANOptimWrapperConstructor('test')

        with self.assertRaises(AssertionError):
            optim_wrapper_dict_builder = SinGANOptimWrapperConstructor(
                self.optim_wrapper_cfg, 'test')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
