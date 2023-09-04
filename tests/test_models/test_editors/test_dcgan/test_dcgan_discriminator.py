# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors.dcgan import DCGANDiscriminator
from mmagic.registry import MODELS


class TestDCGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((2, 3, 32, 32))
        cls.default_config = dict(
            type='DCGANDiscriminator',
            input_scale=32,
            output_scale=4,
            out_channels=5)

    def test_dcgan_discriminator(self):
        # test default setting with builder
        d = MODELS.build(self.default_config)
        pred = d(self.input_tensor)
        assert pred.shape == (2, 5)
        assert d.num_downsamples == 3
        assert len(d.downsamples) == 3
        assert not d.downsamples[0].with_norm
        assert not d.output_layer.with_norm
        assert not d.output_layer.with_activation
        assert isinstance(d.downsamples[1].activate, torch.nn.LeakyReLU)
        assert isinstance(d.downsamples[1].norm, torch.nn.BatchNorm2d)

        # sanity check for args with cpu model
        d = DCGANDiscriminator(input_scale=64, output_scale=8, out_channels=2)
        assert d.input_scale == 64 and d.output_scale == 8
        assert d.num_downsamples == 3
        assert d.out_channels == 2
        pred = d(torch.randn((1, 3, 64, 64)))
        assert pred.shape == (1, 50)

        with pytest.raises(TypeError):
            _ = DCGANDiscriminator(32, 4, 2, pretrained=dict())

        # check for cuda
        if not torch.cuda.is_available():
            return

        # test default setting with builder on GPU
        d = MODELS.build(self.default_config).cuda()
        pred = d(self.input_tensor.cuda())
        assert pred.shape == (2, 5)
        assert d.num_downsamples == 3
        assert len(d.downsamples) == 3
        assert not d.downsamples[0].with_norm
        assert not d.output_layer.with_norm
        assert not d.output_layer.with_activation
        assert isinstance(d.downsamples[1].activate, torch.nn.LeakyReLU)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
