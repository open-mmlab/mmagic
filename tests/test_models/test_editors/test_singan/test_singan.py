# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import MessageHub

from mmagic.engine import SinGANOptimWrapperConstructor
from mmagic.models import SinGAN
from mmagic.utils import register_all_modules

register_all_modules()


class TestSinGAN:

    @classmethod
    def setup_class(cls):
        cls.generator = dict(
            type='SinGANMultiScaleGenerator',
            in_channels=3,
            out_channels=3,
            num_scales=3)

        cls.disc = dict(
            type='SinGANMultiScaleDiscriminator', in_channels=3, num_scales=3)

        cls.data_preprocessor = dict(
            type='DataPreprocessor', non_image_keys=['input_sample'])
        cls.noise_weight_init = 0.1
        cls.curr_scale = -1
        cls.iters_per_scale = 2
        cls.lr_scheduler_args = dict(milestones=[1600], gamma=0.1)

        cls.data_batch = dict(
            inputs=dict(
                real_scale0=torch.randn(1, 3, 25, 25),
                real_scale1=torch.randn(1, 3, 30, 30),
                real_scale2=torch.randn(1, 3, 32, 32),
            ))
        cls.data_batch['inputs']['input_sample'] = torch.zeros_like(
            cls.data_batch['inputs']['real_scale0'])

        cls.optim_wrapper_cfg = dict(
            generator=dict(
                optimizer=dict(type='Adam', lr=0.0005, betas=(0.5, 0.999))),
            discriminator=dict(
                optimizer=dict(type='Adam', lr=0.0005, betas=(0.5, 0.999))))

    def test_singan_cpu(self):
        message_hub = MessageHub.get_instance('singan-test')
        message_hub.update_info('iter', 0)

        singan = SinGAN(
            self.generator,
            self.disc,
            num_scales=3,
            data_preprocessor=self.data_preprocessor,
            noise_weight_init=self.noise_weight_init,
            iters_per_scale=self.iters_per_scale,
            lr_scheduler_args=self.lr_scheduler_args)
        optim_wrapper_dict_builder = SinGANOptimWrapperConstructor(
            self.optim_wrapper_cfg)
        optim_wrapper_dict = optim_wrapper_dict_builder(singan)

        for i in range(6):
            singan.train_step(self.data_batch, optim_wrapper_dict)
            message_hub.update_info('iter', message_hub.get_info('iter') + 1)
            outputs = singan.forward(dict(num_batches=1), None)

            img = torch.stack([out.fake_img.data for out in outputs], dim=0)
            if i in [0, 1]:
                assert singan.curr_stage == 0
                assert img.shape[-2:] == (25, 25)
            elif i in [2, 3]:
                assert singan.curr_stage == 1
                assert img.shape[-2:] == (30, 30)
            elif i in [4, 5]:
                assert singan.curr_stage == 2
                assert img.shape[-2:] == (32, 32)

            outputs = singan.forward(
                dict(num_batches=1, get_prev_res=True), None)
            assert all([hasattr(out, 'prev_res_list') for out in outputs])

        # test forward singan with ema
        singan = SinGAN(
            self.generator,
            self.disc,
            num_scales=3,
            data_preprocessor=self.data_preprocessor,
            noise_weight_init=self.noise_weight_init,
            iters_per_scale=self.iters_per_scale,
            lr_scheduler_args=self.lr_scheduler_args,
            ema_confg=dict(type='ExponentialMovingAverage'))
        optim_wrapper_dict_builder = SinGANOptimWrapperConstructor(
            self.optim_wrapper_cfg)
        optim_wrapper_dict = optim_wrapper_dict_builder(singan)

        for i in range(6):
            singan.train_step(self.data_batch, optim_wrapper_dict)
            message_hub.update_info('iter', message_hub.get_info('iter') + 1)

            outputs = singan.forward(
                dict(num_batches=1, sample_model='ema/orig'), None)

            img = torch.stack([out.orig.fake_img.data for out in outputs],
                              dim=0)
            img_ema = torch.stack([out.ema.fake_img.data for out in outputs],
                                  dim=0)
            if i in [0, 1]:
                assert singan.curr_stage == 0
                assert img.shape[-2:] == (25, 25)
                assert img_ema.shape[-2:] == (25, 25)
            elif i in [2, 3]:
                assert singan.curr_stage == 1
                assert img.shape[-2:] == (30, 30)
                assert img_ema.shape[-2:] == (30, 30)
            elif i in [4, 5]:
                assert singan.curr_stage == 2
                assert img.shape[-2:] == (32, 32)
                assert img_ema.shape[-2:] == (32, 32)

            outputs = singan.forward(
                dict(
                    num_batches=1, sample_model='ema/orig', get_prev_res=True),
                None)

            assert all([hasattr(out.orig, 'prev_res_list') for out in outputs])
            assert all([hasattr(out.ema, 'prev_res_list') for out in outputs])


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
