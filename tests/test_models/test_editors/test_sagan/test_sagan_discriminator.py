# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmedit.models.editors.sagan import ProjDiscriminator
from mmedit.registry import MODULES


class TestSNGANPROJDiscriminator(object):

    @classmethod
    def setup_class(cls):
        cls.x = torch.randn((2, 3, 32, 32))
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='ProjDiscriminator',
            input_scale=32,
            num_classes=10,
            input_channels=3)

    def test_sngan_proj_discriminator(self):

        # test default setting with builder
        d = MODULES.build(self.default_config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different input_scale
        config = deepcopy(self.default_config)
        config['input_scale'] = 64
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x, self.label)
        assert score.shape == (2, 1)

        # test num_classes == 0 (w/o proj_y)
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 128
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different channels_cfg --> list
        config = deepcopy(self.default_config)
        config['channels_cfg'] = [1, 1, 1]
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different channels_cfg --> dict
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {32: [1, 1, 1], 64: [2, 4, 8, 16]}
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different channels_cfg --> error (key not find)
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {64: [2, 4, 8, 16]}
        with pytest.raises(KeyError):
            d = MODULES.build(config)

        # test different channels_cfg --> error (type not match)
        config = deepcopy(self.default_config)
        config['channels_cfg'] = '1234'
        with pytest.raises(ValueError):
            d = MODULES.build(config)

        # test different downsample_cfg --> list
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = [True, False, False]
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different downsample_cfg --> dict
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = {
            32: [True, False, False],
            64: [True, True, True, True]
        }
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different downsample_cfg --> error (key not find)
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = {64: [True, True, True, True]}
        with pytest.raises(KeyError):
            d = MODULES.build(config)

        # test different downsample_cfg --> error (type not match)
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = '1234'
        with pytest.raises(ValueError):
            d = MODULES.build(config)

        # test downsample_cfg and channels_cfg not match
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = [True, False, False]
        config['channels_cfg'] = [1, 1, 1, 1]
        with pytest.raises(ValueError):
            d = MODULES.build(config)

        # test different act_cfg
        config = deepcopy(self.default_config)
        config['act_cfg'] = dict(type='Sigmoid')
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different with_spectral_norm
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = False
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different init_cfg --> studio
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='studio')
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different init_cfg --> BigGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='biggan')
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different init_cfg --> sngan
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan-proj')
        d = MODULES.build(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different init_cfg --> raise error
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='wgan-gp')
        with pytest.raises(NotImplementedError):
            d = MODULES.build(config)

        # test pretrained --> raise error
        config = deepcopy(self.default_config)
        config['pretrained'] = 42
        with pytest.raises(TypeError):
            d = MODULES.build(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_sngan_proj_discriminator_cuda(self):

        # test default setting with builder
        d = MODULES.build(self.default_config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different input_scale
        config = deepcopy(self.default_config)
        config['input_scale'] = 64
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        x = torch.randn((2, 3, 64, 64)).cuda()
        score = d(x, self.label.cuda())
        assert score.shape == (2, 1)

        # test num_classes == 0 (w/o proj_y)
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 128
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different channels_cfg --> list
        config = deepcopy(self.default_config)
        config['channels_cfg'] = [1, 1, 1]
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different channels_cfg --> dict
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {32: [1, 1, 1], 64: [2, 4, 8, 16]}
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different downsample_cfg --> list
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = [True, False, False]
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different downsample_cfg --> dict
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = {
            32: [True, False, False],
            64: [True, True, True, True]
        }
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different act_cfg
        config = deepcopy(self.default_config)
        config['act_cfg'] = dict(type='Sigmoid')
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different with_spectral_norm
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = False
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different init_cfg --> BigGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='biggan')
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different init_cfg --> sngan
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan-proj')
        d = MODULES.build(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)
