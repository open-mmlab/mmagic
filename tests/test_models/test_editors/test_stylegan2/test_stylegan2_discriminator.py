# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from mmengine import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.models.editors.stylegan2 import (ADAAug, ADAStyleGAN2Discriminator,
                                             StyleGAN2Discriminator)
from mmagic.models.utils import get_module_device


@pytest.mark.skipif(
    ('win' in platform.system().lower() and 'cu' in torch.__version__)
    or not torch.cuda.is_available(),
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

        cfg = deepcopy(self.default_cfg)
        cfg['cond_size'] = 5
        cfg['cond_mapping_channels'] = 16

        d = StyleGAN2Discriminator(**cfg)
        score = d(img, torch.randn(2, 5))
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_stylegan2_disc_cuda(self):
        d = StyleGAN2Discriminator(**self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)

        cfg = deepcopy(self.default_cfg)
        cfg['cond_size'] = 5
        cfg['cond_mapping_channels'] = 16

        d = StyleGAN2Discriminator(**cfg).cuda()
        score = d(img, torch.randn(2, 5).cuda())
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


class TestStyleGANv2AdaDisc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_size=64, data_aug=dict(type='ADAAug'), channel_multiplier=1)

    @pytest.mark.skipif(
        digit_version(TORCH_VERSION) <= digit_version('1.6.0'),
        reason='torch version lower than 1.7.0 does not have `torch.exp2` api')
    def test_styleganv2_ada(self):
        disc = ADAStyleGAN2Discriminator(**self.default_cfg)
        assert hasattr(disc, 'ada_aug')
        img = torch.randn(2, 3, 64, 64)
        score = disc(img)
        assert score.shape == (2, 1)

        cfg = deepcopy(self.default_cfg)
        cfg['data_aug'] = None
        disc = ADAStyleGAN2Discriminator(**cfg)
        assert not hasattr(disc, 'ada_aug')
        img = torch.randn(2, 3, 64, 64)
        score = disc(img)
        assert score.shape == (2, 1)


@pytest.mark.skipif(
    digit_version(TORCH_VERSION) <= digit_version('1.6.0'),
    reason='torch version lower than 1.7.0 does not have `torch.exp2` api')
def test_ada_pipeline():
    ada = ADAAug()
    ada.update(0, 2)
    ada.update(1, 2)
    ada.update(2, 2)
    assert (ada.log_buffer == 0).all()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
