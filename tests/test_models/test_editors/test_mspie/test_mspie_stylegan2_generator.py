# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from mmagic.models.editors.mspie import MSStyleGANv2Generator
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()


@MODELS.register_module()
class MockHeadPosEncoding(nn.Module):

    def __init__(self):
        super().__init__()


class TestMSStyleGAN2:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(out_size=32, style_channels=16)

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_msstylegan2_cpu(self):

        # test normal forward
        cfg = deepcopy(self.default_cfg)
        g = MSStyleGANv2Generator(**cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        # set mix_prob as 1.0 and 0 to force cover lines
        cfg = deepcopy(self.default_cfg)
        cfg['mix_prob'] = 1
        g = MSStyleGANv2Generator(**cfg)
        res = g(torch.randn, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        # test style-mixing with inject_inde is passed
        res = g(torch.randn, num_batches=2, inject_index=0)
        assert res.shape == (2, 3, 32, 32)

        cfg = deepcopy(self.default_cfg)
        cfg['mix_prob'] = 0
        g = MSStyleGANv2Generator(**cfg)
        res = g(torch.randn, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        cfg = deepcopy(self.default_cfg)
        cfg['mix_prob'] = 1
        g = MSStyleGANv2Generator(**cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        cfg = deepcopy(self.default_cfg)
        cfg['mix_prob'] = 0
        g = MSStyleGANv2Generator(**cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        # test truncation less than 1
        res = g(None, num_batches=2, truncation=0.5)
        assert res.shape == (2, 3, 32, 32)

        # test chosen scale
        res = g(None, num_batches=2, chosen_scale=2, randomize_noise=False)
        print(res.shape)

        # test injected_noise is not None
        injected_noise = g.make_injected_noise()
        res = g(None, num_batches=2, injected_noise=injected_noise)

        # test return noise is True
        res = g(None, num_batches=2, return_noise=True)
        assert isinstance(res, dict)

        # test chosen_scale is Tuple
        res = g(None, num_batches=2, chosen_scale=(0, 2))
        print(res.shape)

    def test_train(self):
        cfg = deepcopy(self.default_cfg)
        g = MSStyleGANv2Generator(**cfg)
        # test train -> train
        g.train()
        assert g.training
        assert g.default_style_mode == g._default_style_mode
        # test train -> eval
        g.eval()
        assert not g.training
        assert g.default_style_mode == g.eval_style_mode
        # test eval -> eval
        g.eval()
        assert not g.training
        assert g.default_style_mode == g.eval_style_mode
        # test train -> train
        g.train()
        assert g.training
        assert g.default_style_mode == g._default_style_mode

    # def test_make_injected_noise(self):
    #     cfg = deepcopy(self.default_cfg)
    #     g = MSStyleGANv2Generator(**cfg)
    #     pass

    def test_mean_latent(self):
        cfg = deepcopy(self.default_cfg)
        g = MSStyleGANv2Generator(**cfg)
        mean_latent = g.get_mean_latent(num_samples=4, bs_per_repeat=2)
        assert mean_latent.shape == (1, 16)

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_head_pos_encoding(self):
        cfg = deepcopy(self.default_cfg)
        g = MSStyleGANv2Generator(**cfg, head_pos_encoding=dict(type='CSG'))
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        g = MSStyleGANv2Generator(
            **cfg, head_pos_encoding=dict(type='CSG'), interp_head=True)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        g = MSStyleGANv2Generator(
            **cfg, head_pos_encoding=dict(type='MockHeadPosEncoding'))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
