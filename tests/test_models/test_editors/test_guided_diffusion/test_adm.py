# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmagic.models import AblatedDiffusionModel
from mmagic.utils import register_all_modules

register_all_modules()


class TestAdm(TestCase):

    def setup_class(cls):
        # test init
        cls.model = AblatedDiffusionModel(
            data_preprocessor=dict(type='DataPreprocessor'),
            unet=dict(
                type='DenoisingUnet',
                image_size=64,
                in_channels=3,
                base_channels=192,
                resblocks_per_downsample=3,
                attention_res=(32, 16, 8),
                norm_cfg=dict(type='GN32', num_groups=32),
                dropout=0.1,
                num_classes=1000,
                use_fp16=False,
                resblock_updown=True,
                attention_cfg=dict(
                    type='MultiHeadAttentionBlock',
                    num_heads=4,
                    num_head_channels=64,
                    use_new_attention_order=True),
                use_scale_shift_norm=True),
            diffusion_scheduler=dict(
                type='EditDDIMScheduler',
                variance_type='learned_range',
                beta_schedule='squaredcos_cap_v2'),
            rgb2bgr=True,
            use_fp16=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_infer(self):
        # test no label infer
        self.model.cuda()
        samples = self.model.infer(
            init_image=None,
            batch_size=1,
            num_inference_steps=5,
            labels=None,
            classifier_scale=0.0,
            show_progress=False)['samples']
        assert samples.shape == (1, 3, 64, 64)
        # test classifier guidance
        samples = self.model.infer(
            init_image=None,
            batch_size=1,
            num_inference_steps=5,
            labels=333,
            classifier_scale=1.0,
            show_progress=False)['samples']
        assert samples.shape == (1, 3, 64, 64)
        # test with ddpm scheduler
        scheduler_kwargs = dict(
            type='EditDDPMScheduler',
            variance_type='learned_range',
            num_train_timesteps=5)
        # test no label infer
        samples = self.model.infer(
            scheduler_kwargs=scheduler_kwargs,
            init_image=None,
            batch_size=1,
            num_inference_steps=5,
            labels=None,
            classifier_scale=0.0,
            show_progress=False)['samples']
        assert samples.shape == (1, 3, 64, 64)
        # test classifier guidance
        samples = self.model.infer(
            scheduler_kwargs=scheduler_kwargs,
            init_image=None,
            batch_size=1,
            num_inference_steps=5,
            labels=333,
            classifier_scale=1.0,
            show_progress=False)['samples']
        assert samples.shape == (1, 3, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
