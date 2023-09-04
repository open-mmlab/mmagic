# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.deblurganv2 import DeblurGanV2Generator
from mmagic.models.editors.deblurganv2.deblurganv2_generator import \
    FPNMobileNet
from mmagic.registry import MODELS


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
class TestDeblurGanv2Generator(object):

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((1, 3, 256, 256))
        cls.default_config = dict(
            type='DeblurGanV2Generator',
            backbone='FPNMobileNet',
            norm_layer='instance',
            output_ch=3,
            num_filter=64,
            num_filter_fpn=128)

    def test_deblurganv2_generator(self):

        # test default setting with builder
        g = MODELS.build(self.default_config)
        assert isinstance(g, FPNMobileNet)

        # check forward function
        img = g(self.input_tensor)
        assert img.shape == (1, 3, 256, 256)
        img = g(torch.randn(4, 3, 256, 256))
        assert img.shape == (4, 3, 256, 256)
        with pytest.raises(TypeError):
            _ = DeblurGanV2Generator()

        # sanity check for args with cpu model
        g = DeblurGanV2Generator(
            backbone='FPNMobileNet',
            norm_layer='instance',
            output_ch=3,
            num_filter=64,
            num_filter_fpn=128)
        img = g(self.input_tensor)
        assert img.shape == (1, 3, 256, 256)

        # check for cuda
        if not torch.cuda.is_available():
            return

        g = MODELS.build(self.default_config).cuda()
        assert isinstance(g, DeblurGanV2Generator)
        g = DeblurGanV2Generator(
            backbone='FPNMobileNet',
            norm_layer='instance',
            output_ch=3,
            num_filter=64,
            num_filter_fpn=128).cuda()
        img = g(self.input_tensor)
        assert img.shape == (1, 3, 256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
