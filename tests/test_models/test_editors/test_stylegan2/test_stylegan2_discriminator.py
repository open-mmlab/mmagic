# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch.nn as nn

from mmedit.models.editors.stylegan2 import StyleGAN2Discriminator
from mmedit.models.utils import get_module_device


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
class TestStyleGANv2Disc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=64, channel_multiplier=1)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_stylegan2_disc_cpu(self):
        d = StyleGAN2Discriminator(**self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_stylegan2_disc_cuda(self):
        d = StyleGAN2Discriminator(**self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)


def test_get_module_device_cpu():
    device = get_module_device(nn.Conv2d(3, 3, 3, 1, 1))
    assert device == torch.device('cpu')

    # The input module should contain parameters.
    with pytest.raises(ValueError):
        get_module_device(nn.Flatten())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
def test_get_module_device_cuda():
    module = nn.Conv2d(3, 3, 3, 1, 1).cuda()
    device = get_module_device(module)
    assert device == next(module.parameters()).get_device()

    # The input module should contain parameters.
    with pytest.raises(ValueError):
        get_module_device(nn.Flatten().cuda())
