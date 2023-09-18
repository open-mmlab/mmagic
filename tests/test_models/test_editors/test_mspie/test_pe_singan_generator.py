# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors.mspie import SinGANMSGeneratorPE


class TestSinGANPEGen:

    @classmethod
    def setup_class(cls):
        cls.default_args = dict(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            num_layers=3,
            base_channels=32,
            num_scales=3,
            min_feat_channels=16)

        cls.fixed_noises = [
            torch.randn(1, 1, 8, 8),
            torch.randn(1, 3, 10, 10),
            torch.randn(1, 3, 12, 12),
            torch.randn(1, 3, 16, 16)
        ]
        cls.input_sample = torch.zeros((1, 3, 8, 8))
        cls.noise_weights = [1., 0.5, 0.5, 0.5]

    def test_singan_gen_pe(self):
        gen = SinGANMSGeneratorPE(**self.default_args)
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

        gen = SinGANMSGeneratorPE(
            padding_mode='reflect', **self.default_args)  # noqa
        res = gen(self.input_sample, self.fixed_noises, self.noise_weights,
                  'rand', 2)
        assert res.shape == (1, 3, 12, 12)

        with pytest.raises(NotImplementedError):
            _ = SinGANMSGeneratorPE(
                padding_mode='circular', **self.default_args)

        gen = SinGANMSGeneratorPE(
            padding=1, pad_at_head=False, **self.default_args)
        res = gen(self.input_sample, self.fixed_noises, self.noise_weights,
                  'rand', 2)
        assert res.shape == (1, 3, 12, 12)

        gen = SinGANMSGeneratorPE(
            pad_at_head=True, interp_pad=True, **self.default_args)
        res = gen(self.input_sample, self.fixed_noises, self.noise_weights,
                  'rand', 2)
        assert res.shape == (1, 3, 12, 12)

        gen = SinGANMSGeneratorPE(
            positional_encoding=dict(
                type='SPE2d', embedding_dim=4, padding_idx=0),
            allow_no_residual=True,
            first_stage_in_channels=8,
            **self.default_args)
        res = gen(self.input_sample, self.fixed_noises, self.noise_weights,
                  'rand', 2)
        assert res.shape == (1, 3, 12, 12)

        gen = SinGANMSGeneratorPE(
            positional_encoding=dict(type='CSG2d'),
            allow_no_residual=True,
            first_stage_in_channels=2,
            **self.default_args)
        res = gen(self.input_sample, self.fixed_noises, self.noise_weights,
                  'rand', 2)
        assert res.shape == (1, 3, 12, 12)

        gen = SinGANMSGeneratorPE(
            interp_pad=True, noise_with_pad=True, **self.default_args)
        res = gen(None, self.fixed_noises, self.noise_weights, 'rand', 2)
        assert res.shape == (1, 3, 6, 6)

        gen = SinGANMSGeneratorPE(
            interp_pad=True, noise_with_pad=False, **self.default_args)
        res = gen(None, self.fixed_noises, self.noise_weights, 'rand', 2)
        assert res.shape == (1, 3, 12, 12)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
