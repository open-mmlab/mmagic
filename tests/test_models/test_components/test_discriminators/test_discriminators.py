# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest
import torch

from mmedit.models import build_component


def test_ttsr_dict():
    cfg = dict(type='TTSRDiscriminator', in_channels=3, in_size=160)
    net = build_component(cfg)
    net.init_weights(pretrained=None)
    # cpu
    inputs = torch.rand((2, 3, 160, 160))
    output = net(inputs)
    assert output.shape == (2, 1)
    # gpu
    if torch.cuda.is_available():
        net.init_weights(pretrained=None)
        net = net.cuda()
        output = net(inputs.cuda())
        assert output.shape == (2, 1)

    # pretrained should be str or None
    with pytest.raises(TypeError):
        net.init_weights(pretrained=[1])


def test_patch_discriminator():
    # color, BN
    cfg = dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='normal', gain=0.02))
    net = build_component(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 3, 64, 64)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 6, 6)
    # gpu
    if torch.cuda.is_available():
        net.init_weights(pretrained=None)
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 6, 6)

    # pretrained should be str or None
    with pytest.raises(TypeError):
        net.init_weights(pretrained=[1])

    # gray, IN
    cfg = dict(
        type='PatchDiscriminator',
        in_channels=1,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02))
    net = build_component(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 1, 64, 64)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 6, 6)
    # gpu
    if torch.cuda.is_available():
        net.init_weights(pretrained=None)
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 6, 6)

    # pretrained should be str or None
    with pytest.raises(TypeError):
        net.init_weights(pretrained=[1])

    # test norm_cfg assertions
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['norm_cfg'] = None
    with pytest.raises(AssertionError):
        _ = build_component(bad_cfg)
    bad_cfg['norm_cfg'] = dict(tp='BN')
    with pytest.raises(AssertionError):
        _ = build_component(bad_cfg)


def test_smpatch_discriminator():
    # color, BN
    cfg = dict(
        type='SoftMaskPatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        with_spectral_norm=True)
    net = build_component(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 3, 64, 64)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 6, 6)
    # gpu
    if torch.cuda.is_available():
        net.init_weights(pretrained=None)
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 6, 6)

    # pretrained should be str or None
    with pytest.raises(TypeError):
        net.init_weights(pretrained=[1])

    # gray, IN
    cfg = dict(
        type='SoftMaskPatchDiscriminator',
        in_channels=1,
        base_channels=64,
        num_conv=3,
        with_spectral_norm=True)
    net = build_component(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 1, 64, 64)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 6, 6)
    # gpu
    if torch.cuda.is_available():
        net.init_weights(pretrained=None)
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 6, 6)

    # pretrained should be str or None
    with pytest.raises(TypeError):
        net.init_weights(pretrained=[1])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).

    Returns:
        imgs: (Tensor): Images in FloatTensor with desired shapes.
    """
    imgs = np.random.random(input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs
