# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.singan import SinGANMultiScaleDiscriminator


class TestSinGANDisc:

    @classmethod
    def setup_class(cls):
        cls.default_args = dict(
            in_channels=3,
            kernel_size=3,
            padding=0,
            num_layers=3,
            base_channels=32,
            num_scales=3,
            min_feat_channels=16)

    def test_singan_disc(self):
        disc = SinGANMultiScaleDiscriminator(**self.default_args)
        img = torch.randn(1, 3, 24, 24)
        res = disc(img, 2)
        assert res.shape[0] == 1


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
