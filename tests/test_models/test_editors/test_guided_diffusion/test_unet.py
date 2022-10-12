# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmgen.models import DenoisingUnet, build_module
from mmedit.models import 

class TestDDPM:

    @classmethod
    def setup_class(cls):
        cls.denoising_cfg = dict(
            type='DenoisingUnet',
            image_size=32,
            in_channels=3,
            base_channels=128,
            resblocks_per_downsample=3,
            attention_res=[16, 8],
            use_scale_shift_norm=True,
            dropout=0,
            num_heads=4)
        cls.x_t = torch.randn(2, 3, 32, 32)
        cls.label = torch.randint(0, 10, (2, ))

    def test_denoising_cpu(self):
        # test default config
        denoising = build_module(self.denoising_cfg)
        assert isinstance(denoising, DenoisingUnet)
        output_dict = denoising(self.x_t, self.timesteps, return_noise=True)
        assert 'outputs' in output_dict

        # test class-conditional denoising
        output_dict = denoising(self.x_t, self.timesteps, label = self.label, return_noise=True)
        assert 'outputs' in output_dict
