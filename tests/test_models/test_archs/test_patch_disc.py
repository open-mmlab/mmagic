# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()


def test_patch_discriminator():
    # color, BN
    cfg = dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='normal', gain=0.02))
    net = MODELS.build(cfg)
    # cpu
    input_shape = (1, 3, 64, 64)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 6, 6)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 6, 6)

    # gray, IN
    cfg = dict(
        type='PatchDiscriminator',
        in_channels=1,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02))
    net = MODELS.build(cfg)
    # cpu
    input_shape = (1, 1, 64, 64)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 6, 6)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 6, 6)

    # test norm_cfg assertions
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['norm_cfg'] = None
    with pytest.raises(AssertionError):
        _ = MODELS.build(bad_cfg)
    bad_cfg['norm_cfg'] = dict(tp='BN')
    with pytest.raises(AssertionError):
        _ = MODELS.build(bad_cfg)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
