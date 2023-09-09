# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.deblurganv2 import DeblurGanV2Discriminator
from mmagic.models.editors.deblurganv2.deblurganv2_discriminator import \
    DoubleGan
from mmagic.registry import MODELS


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
class TestDeblurGanv2Discriminator(object):

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((1, 3, 256, 256))
        cls.default_config = dict(
            type='DeblurGanV2Discriminator',
            backbone='DoubleGan',
            norm_layer='instance',
            d_layers=3)

    def test_deblurganv2_discriminator(self):
        # test default setting with builder
        d = MODELS.build(self.default_config)
        assert isinstance(d, DoubleGan)
        pred = d(self.input_tensor)
        assert isinstance(pred, list)
        assert len(pred) == 2
        assert d.full_gan
        assert d.patch_gan
        with pytest.raises(TypeError):
            _ = DeblurGanV2Discriminator()

        # sanity check for args with cpu model
        d = DeblurGanV2Discriminator(
            backbone='DoubleGan', norm_layer='instance', d_layers=3)
        pred = d(torch.randn((1, 3, 256, 256)))
        assert d.full_gan
        assert d.patch_gan
        assert isinstance(pred, list)
        assert len(pred) == 2

        # check for cuda
        if not torch.cuda.is_available():
            return

        # test default setting with builder on GPU
        d = MODELS.build(self.default_config).cuda()
        pred = d(self.input_tensor.cuda())
        assert d.full_gan
        assert d.patch_gan
        assert isinstance(pred, list)
        assert len(pred) == 2


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
