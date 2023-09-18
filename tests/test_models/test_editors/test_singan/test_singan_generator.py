# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.singan import SinGANMultiScaleGenerator


class TestSinGANGen:

    @classmethod
    def setup_class(cls):
        cls.default_args = dict(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            padding=0,
            num_layers=3,
            base_channels=32,
            num_scales=3,
            min_feat_channels=16)

        cls.fixed_noises = [
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 10, 10),
            torch.randn(1, 3, 12, 12),
            torch.randn(1, 3, 16, 16)
        ]
        cls.input_sample = torch.zeros_like(cls.fixed_noises[0])
        cls.noise_weights = [1., 0.5, 0.5, 0.5]

    def test_singan_gen(self):
        gen = SinGANMultiScaleGenerator(**self.default_args)
        res = gen(self.input_sample, self.fixed_noises, self.noise_weights,
                  'rand', 2)
        assert res.shape == (1, 3, 12, 12)

        output = gen(
            self.input_sample,
            self.fixed_noises,
            self.noise_weights,
            'rand',
            2,
            get_prev_res=True)

        assert output['prev_res_list'][0].shape == (1, 3, 8, 8)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
