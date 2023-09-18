# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmagic.registry import MODELS


class TestSNGANGenResBlock(object):

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn(2, 16, 5, 5)
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='SNGANGenResBlock',
            num_classes=10,
            in_channels=16,
            out_channels=16,
            use_cbn=True,
            use_norm_affine=False,
            norm_cfg=dict(type='BN'),
            upsample_cfg=dict(type='nearest', scale_factor=2),
            upsample=True,
            init_cfg=dict(type='BigGAN'))

    def test_snganGenResBlock(self):

        # test default config
        block = MODELS.build(self.default_config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 10, 10)

        # test no upsample config and no learnable sc
        config = deepcopy(self.default_config)
        config['upsample'] = False
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 5, 5)

        # test learnable shortcut + w/o upsample
        config = deepcopy(self.default_config)
        config['out_channels'] = 32
        config['upsample'] = False
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 32, 5, 5)

        # test init_cfg + w/o learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan')
        config['upsample'] = False
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 5, 5)

        # test init_cfg == studio + w/o learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='studio')
        config['upsample'] = False
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 5, 5)

        # test init_cfg == sagan + learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 10, 10)

        # test init_cfg == sagan + w/o learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        config['upsample'] = False
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 5, 5)

        # test init_cft --> raise error
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='wgan-gp')
        with pytest.raises(NotImplementedError):
            block = MODELS.build(config)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 10, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_snganGenResBlock_cuda(self):

        # test default config
        block = MODELS.build(self.default_config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 10, 10)

        # test no upsample config and no learnable sc
        config = deepcopy(self.default_config)
        config['upsample'] = False
        block = MODELS.build(config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test init_cfg == studio + w/o learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='studio')
        config['upsample'] = False
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 5, 5)

        # test init_cfg == sagan + learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 10, 10)

        # test init_cfg == sagan + w/o learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        config['upsample'] = False
        block = MODELS.build(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 5, 5)

        # test learnable shortcut + w/o upsample
        config = deepcopy(self.default_config)
        config['out_channels'] = 32
        config['upsample'] = False
        block = MODELS.build(config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 32, 5, 5)

        # test init_cfg + w/o learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan')
        config['upsample'] = False
        block = MODELS.build(config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = MODELS.build(config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 10, 10)


class TestSNDiscResBlock(object):

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn(2, 16, 10, 10)
        cls.default_config = dict(
            type='SNGANDiscResBlock',
            in_channels=16,
            out_channels=16,
            downsample=True,
            init_cfg=dict(type='BigGAN'))

    def test_snganDiscResBlock(self):
        # test default config
        block = MODELS.build(self.default_config)
        out = block(self.input)
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = MODELS.build(config)
        out = block(self.input)
        assert out.shape == (2, 16, 5, 5)

        # test w/o learnabel shortcut + w/o downsample
        config = deepcopy(self.default_config)
        config['downsample'] = False
        config['out_channels'] = 8
        block = MODELS.build(config)
        out = block(self.input)
        assert out.shape == (2, 8, 10, 10)

        # test init cfg + w or w/o downsample
        for init_method in [
                'studio', 'biggan', 'sagan', 'sngan', 'sngan-proj', 'gan-proj'
        ]:
            config = deepcopy(self.default_config)
            config['init_cfg'] = dict(type=init_method)
            config['out_channels'] = 8
            for downsample in [True, False]:
                config['downsample'] = downsample
                block = MODELS.build(config)
                out = block(self.input)
                if downsample:
                    assert out.shape == (2, 8, 5, 5)
                else:
                    assert out.shape == (2, 8, 10, 10)

        # test init_cft --> raise error
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='wgan-gp')
        with pytest.raises(NotImplementedError):
            block = MODELS.build(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_snganDiscResBlock_cuda(self):
        # test default config
        block = MODELS.build(self.default_config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = MODELS.build(config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test w/o learnabel shortcut + w/o downsample
        config = deepcopy(self.default_config)
        config['downsample'] = False
        config['out_channels'] = 8
        block = MODELS.build(config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 8, 10, 10)

        # test init cfg + w or w/o downsample
        for init_method in [
                'studio', 'biggan', 'sagan', 'sngan', 'sngan-proj', 'gan-proj'
        ]:
            config = deepcopy(self.default_config)
            config['init_cfg'] = dict(type=init_method)
            config['out_channels'] = 8
            for downsample in [True, False]:
                config['downsample'] = downsample
                block = MODELS.build(config)
                out = block(self.input)
                if downsample:
                    assert out.shape == (2, 8, 5, 5)
                else:
                    assert out.shape == (2, 8, 10, 10)


class TestSNDiscHeadResBlock(object):

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn(2, 16, 10, 10)
        cls.default_config = dict(
            type='SNGANDiscHeadResBlock',
            in_channels=16,
            out_channels=16,
            init_cfg=dict(type='BigGAN'))

    def test_snganDiscHeadResBlock(self):
        # test default config
        block = MODELS.build(self.default_config)
        out = block(self.input)
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = MODELS.build(config)
        out = block(self.input)
        assert out.shape == (2, 16, 5, 5)

        # test init cfg + w or w/o downsample
        for init_method in [
                'studio', 'biggan', 'sagan', 'sngan', 'sngan-proj', 'gan-proj'
        ]:
            config = deepcopy(self.default_config)
            config['init_cfg'] = dict(type=init_method)
            block = MODELS.build(config)
            out = block(self.input)
            assert out.shape == (2, 16, 5, 5)

        # test init_cft --> raise error
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='wgan-gp')
        with pytest.raises(NotImplementedError):
            block = MODELS.build(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_snganDiscHeadResBlock_cuda(self):
        # test default config
        block = MODELS.build(self.default_config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test init cfg + w or w/o downsample
        for init_method in [
                'studio', 'biggan', 'sagan', 'sngan', 'sngan-proj', 'gan-proj'
        ]:
            config = deepcopy(self.default_config)
            config['init_cfg'] = dict(type=init_method)
            block = MODELS.build(config)
            out = block(self.input)
            assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = MODELS.build(config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 16, 5, 5)


class TestSNConditionalNorm(object):

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn((2, 4, 4, 4))
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='SNConditionNorm',
            in_channels=4,
            num_classes=10,
            use_cbn=True,
            cbn_norm_affine=False,
            init_cfg=dict(type='BigGAN'))

    def test_conditionalNorm(self):
        # test build from default config
        norm = MODELS.build(self.default_config)
        out = norm(self.input, self.label)
        assert out.shape == (2, 4, 4, 4)

        # test w/o use_cbn
        config = deepcopy(self.default_config)
        config['use_cbn'] = False
        norm = MODELS.build(config)
        out = norm(self.input)
        assert out.shape == (2, 4, 4, 4)

        # test num_class < 0 and cbn = False
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        config['use_cbn'] = False
        norm = MODELS.build(config)
        out = norm(self.input)
        assert out.shape == (2, 4, 4, 4)

        # test num_classes <= 0 and cbn = True
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        with pytest.raises(ValueError):
            norm = MODELS.build(config)

        # test IN
        config = deepcopy(self.default_config)
        config['norm_cfg'] = dict(type='IN')
        norm = MODELS.build(config)
        out = norm(self.input, self.label)
        assert out.shape == (2, 4, 4, 4)

        # test sn_style == ajbrock
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        config['sn_style'] = 'ajbrock'
        norm = MODELS.build(config)
        out = norm(self.input, self.label)
        for buffer in ['u0', 'sv0']:
            assert hasattr(norm.weight_embedding, buffer)
            assert hasattr(norm.bias_embedding, buffer)
        assert out.shape == (2, 4, 4, 4)

        # test sn_style == torch
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        config['sn_style'] = 'torch'
        norm = MODELS.build(config)
        out = norm(self.input, self.label)
        for buffer in ['weight_u', 'weight_v', 'weight_orig']:
            assert hasattr(norm.weight_embedding, buffer)
            assert hasattr(norm.bias_embedding, buffer)
        assert out.shape == (2, 4, 4, 4)

        # test sn_style --> raise error
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        config['sn_style'] = 'studio'
        with pytest.raises(NotImplementedError):
            norm = MODELS.build(config)

        # test SyncBN
        # config = deepcopy(self.default_config)
        # config['norm_cfg'] = dict(type='SyncBN')
        # norm = MODELS.build(config)
        # out = norm(self.input, self.label)
        # assert out.shape == (2, 4, 4, 4)

        # test unknown norm type
        config = deepcopy(self.default_config)
        config['norm_cfg'] = dict(type='GN')
        with pytest.raises(ValueError):
            norm = MODELS.build(config)

        # test init_cfg
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan')
        norm = MODELS.build(config)
        out = norm(self.input, self.label)
        assert out.shape == (2, 4, 4, 4)

        # test init_cft --> raise error
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='wgan-gp')
        with pytest.raises(NotImplementedError):
            norm = MODELS.build(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_conditionalNorm_cuda(self):
        # test build from default config
        norm = MODELS.build(self.default_config).cuda()
        out = norm(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 4, 4, 4)

        # test w/o use_cbn
        config = deepcopy(self.default_config)
        config['use_cbn'] = False
        norm = MODELS.build(config).cuda()
        out = norm(self.input.cuda())
        assert out.shape == (2, 4, 4, 4)

        # test num_class < 0 and cbn = False
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        config['use_cbn'] = False
        norm = MODELS.build(config).cuda()
        out = norm(self.input.cuda())
        assert out.shape == (2, 4, 4, 4)

        # test num_classes <= 0 and cbn = True
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        with pytest.raises(ValueError):
            norm = MODELS.build(config)

        # test IN
        config = deepcopy(self.default_config)
        config['norm_cfg'] = dict(type='IN')
        norm = MODELS.build(config).cuda()
        out = norm(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 4, 4, 4)

        # test sn_style == ajbrock
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        config['sn_style'] = 'ajbrock'
        norm = MODELS.build(config)
        out = norm(self.input, self.label)
        for buffer in ['u0', 'sv0']:
            assert hasattr(norm.weight_embedding, buffer)
            assert hasattr(norm.bias_embedding, buffer)
        assert out.shape == (2, 4, 4, 4)

        # test sn_style == torch
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        config['sn_style'] = 'torch'
        norm = MODELS.build(config)
        out = norm(self.input, self.label)
        for buffer in ['weight_u', 'weight_v', 'weight_orig']:
            assert hasattr(norm.weight_embedding, buffer)
            assert hasattr(norm.bias_embedding, buffer)
        assert out.shape == (2, 4, 4, 4)

        # test sn_style --> raise error
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        config['sn_style'] = 'studio'
        with pytest.raises(NotImplementedError):
            norm = MODELS.build(config)

        # test SyncBN
        # config = deepcopy(self.default_config)
        # config['norm_cfg'] = dict(type='SyncBN')
        # norm = MODELS.build(config)
        # out = norm(self.input, self.label)
        # assert out.shape == (2, 4, 4, 4)

        # test unknown norm type
        config = deepcopy(self.default_config)
        config['norm_cfg'] = dict(type='GN')
        with pytest.raises(ValueError):
            norm = MODELS.build(config)

        # test init_cfg
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan')
        norm = MODELS.build(config).cuda()
        out = norm(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 4, 4, 4)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
