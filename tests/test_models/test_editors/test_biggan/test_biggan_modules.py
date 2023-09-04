# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

# yapf:disable
from mmagic.models.editors.biggan import (BigGANConditionBN,
                                          BigGANDeepDiscResBlock,
                                          BigGANDeepGenResBlock,
                                          BigGANDiscResBlock,
                                          BigGANGenResBlock,
                                          SelfAttentionBlock)
from mmagic.registry import MODELS

# yapf:enable


class TestBigGANDeepGenResBlock:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='BigGANDeepGenResBlock',
            in_channels=32,
            out_channels=16,
            dim_after_concat=100,
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(type='nearest', scale_factor=2),
            sn_eps=1e-6,
            bn_eps=1e-5,
            with_spectral_norm=True,
            input_is_label=False,
            auto_sync_bn=True,
            channel_ratio=4)
        cls.x = torch.randn(2, 32, 8, 8)
        cls.y = torch.randn(2, 100)
        cls.label = torch.randint(0, 100, (2, ))

    def test_biggan_deep_gen_res_block(self):
        # test default setting
        module = MODELS.build(self.default_cfg)
        assert isinstance(module, BigGANDeepGenResBlock)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 16, 16)

        # test without upsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(upsample_cfg=None))
        module = MODELS.build(cfg)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 8, 8)

        # test input_is_label
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_is_label=True))
        module = MODELS.build(cfg)
        out = module(self.x, self.label)
        assert out.shape == (2, 16, 16, 16)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_deep_gen_res_block_cuda(self):
        # test default setting
        module = MODELS.build(self.default_cfg).cuda()
        assert isinstance(module, BigGANDeepGenResBlock)
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 16, 16)

        # test without upsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(upsample_cfg=None))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 8, 8)

        # test input_is_label
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_is_label=True))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 16, 16)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 16, 16)


class TestBigGANDeepDiscResBlock:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='BigGANDeepDiscResBlock',
            in_channels=32,
            out_channels=64,
            channel_ratio=4,
            act_cfg=dict(type='ReLU', inplace=False),
            sn_eps=1e-6,
            with_downsample=True,
            with_spectral_norm=True)
        cls.x = torch.randn(2, 32, 16, 16)

    def test_biggan_deep_disc_res_block(self):
        # test default setting
        module = MODELS.build(self.default_cfg)
        assert isinstance(module, BigGANDeepDiscResBlock)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

        # test with_downsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(with_downsample=False))
        module = MODELS.build(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 16, 16)

        # test different channel_ratio
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(channel_ratio=8))
        module = MODELS.build(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_deep_disc_res_block_cuda(self):
        # test default setting
        module = MODELS.build(self.default_cfg).cuda()
        assert isinstance(module, BigGANDeepDiscResBlock)
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 8, 8)

        # test with_downsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(with_downsample=False))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 16, 16)

        # test different channel_ratio
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(channel_ratio=8))
        module = MODELS.build(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 8, 8)


class TestBigGANConditionBN:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='BigGANConditionBN',
            num_features=64,
            linear_input_channels=80)
        cls.x = torch.randn(2, 64, 32, 32)
        cls.y = torch.randn(2, 80)
        cls.label = torch.randint(0, 80, (2, ))

    def test_biggan_condition_bn(self):
        # test default setting
        module = MODELS.build(self.default_cfg)
        assert isinstance(module, BigGANConditionBN)
        out = module(self.x, self.y)
        assert out.shape == (2, 64, 32, 32)

        # test input_is_label
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_is_label=True))
        module = MODELS.build(cfg)
        out = module(self.x, self.label)
        assert out.shape == (2, 64, 32, 32)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg)
        out = module(self.x, self.y)
        assert out.shape == (2, 64, 32, 32)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg)
        out = module(self.x, self.y)
        assert out.shape == (2, 64, 32, 32)

        # test not-implemented sn-style
        with pytest.raises(NotImplementedError):
            cfg = deepcopy(self.default_cfg)
            cfg.update(dict(sn_style='tero'))
            module = MODELS.build(cfg)
            out = module(self.x, self.y)
            assert out.shape == (2, 64, 32, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_condition_bn_cuda(self):
        # test default setting
        module = MODELS.build(self.default_cfg).cuda()
        assert isinstance(module, BigGANConditionBN)
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 64, 32, 32)

        # test input_is_label
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_is_label=True))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda(), self.label.cuda())
        assert out.shape == (2, 64, 32, 32)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 64, 32, 32)

        # test not-implemented sn-style
        with pytest.raises(NotImplementedError):
            cfg = deepcopy(self.default_cfg)
            cfg.update(dict(sn_style='tero'))
            module = MODELS.build(cfg).cuda()
            out = module(self.x.cuda(), self.y.cuda())
            assert out.shape == (2, 64, 32, 32)


class TestSelfAttentionBlock:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(type='SelfAttentionBlock', in_channels=16)
        cls.x = torch.randn(2, 16, 8, 8)

    def test_self_attention_block(self):
        # test default setting
        module = MODELS.build(self.default_cfg)
        assert isinstance(module, SelfAttentionBlock)
        out = module(self.x)
        assert out.shape == (2, 16, 8, 8)

        # test different in_channels
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(in_channels=10))
        module = MODELS.build(cfg)
        x = torch.randn(2, 10, 8, 8)
        out = module(x)
        assert out.shape == (2, 10, 8, 8)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg)
        out = module(self.x)
        assert out.shape == (2, 16, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_self_attention_block_cuda(self):
        # test default setting
        module = MODELS.build(self.default_cfg).cuda()
        assert isinstance(module, SelfAttentionBlock)
        out = module(self.x.cuda())
        assert out.shape == (2, 16, 8, 8)

        # test different in_channels
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(in_channels=10))
        module = MODELS.build(cfg).cuda()
        x = torch.randn(2, 10, 8, 8).cuda()
        out = module(x)
        assert out.shape == (2, 10, 8, 8)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda())
        assert out.shape == (2, 16, 8, 8)


class TestBigGANGenResBlock:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='BigGANGenResBlock',
            in_channels=32,
            out_channels=16,
            dim_after_concat=100,
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(type='nearest', scale_factor=2),
            sn_eps=1e-6,
            with_spectral_norm=True,
            input_is_label=False,
            auto_sync_bn=True)
        cls.x = torch.randn(2, 32, 8, 8)
        cls.y = torch.randn(2, 100)
        cls.label = torch.randint(0, 100, (2, ))

    def test_biggan_gen_res_block(self):
        # test default setting
        module = MODELS.build(self.default_cfg)
        assert isinstance(module, BigGANGenResBlock)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 16, 16)

        # test without upsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(upsample_cfg=None))
        module = MODELS.build(cfg)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 8, 8)

        # test input_is_label
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_is_label=True))
        module = MODELS.build(cfg)
        out = module(self.x, self.label)
        assert out.shape == (2, 16, 16, 16)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_gen_res_block_cuda(self):
        # test default setting
        module = MODELS.build(self.default_cfg).cuda()
        assert isinstance(module, BigGANGenResBlock)
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 16, 16)

        # test without upsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(upsample_cfg=None))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 8, 8)

        # test input_is_label
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_is_label=True))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 16, 16)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 16, 16)


class TestBigGANDiscResBlock:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='BigGANDiscResBlock',
            in_channels=32,
            out_channels=64,
            act_cfg=dict(type='ReLU', inplace=False),
            sn_eps=1e-6,
            with_downsample=True,
            with_spectral_norm=True,
            is_head_block=False)
        cls.x = torch.randn(2, 32, 16, 16)

    def test_biggan_disc_res_block(self):
        # test default setting
        module = MODELS.build(self.default_cfg)
        assert isinstance(module, BigGANDiscResBlock)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

        # test with_downsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(with_downsample=False))
        module = MODELS.build(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 16, 16)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_disc_res_block_cuda(self):
        # test default setting
        module = MODELS.build(self.default_cfg).cuda()
        assert isinstance(module, BigGANDiscResBlock)
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 8, 8)

        # test with_downsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(with_downsample=False))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 16, 16)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = MODELS.build(cfg).cuda()
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 8, 8)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
