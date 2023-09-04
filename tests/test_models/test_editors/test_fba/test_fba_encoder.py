# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmagic.models.editors import FBAResnetDilated


def _demo_inputs(input_shape=(1, 4, 64, 64)):
    """Create a superset of inputs needed to run encoder.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 4, 64, 64).
    """
    img = np.random.random(input_shape).astype(np.float32)
    img = torch.from_numpy(img)

    return img


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def test_fba_encoder():
    """Test FBA encoder."""

    with pytest.raises(KeyError):
        # ResNet depth should be in [18, 34, 50, 101, 152]
        FBAResnetDilated(
            20,
            in_channels=11,
            stem_channels=64,
            base_channels=64,
        )

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        FBAResnetDilated(
            50,
            in_channels=11,
            stem_channels=64,
            base_channels=64,
            num_stages=0)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        FBAResnetDilated(
            50,
            in_channels=11,
            stem_channels=64,
            base_channels=64,
            num_stages=5)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations) == num_stages
        FBAResnetDilated(
            50,
            in_channels=11,
            stem_channels=64,
            base_channels=64,
            strides=(1, ),
            dilations=(1, 1),
            num_stages=3)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = FBAResnetDilated(
            50,
            in_channels=11,
            stem_channels=64,
            base_channels=64,
        )
        model.init_weights(pretrained=233)

    model = FBAResnetDilated(
        depth=50,
        in_channels=11,
        stem_channels=64,
        base_channels=64,
        conv_cfg=dict(type='ConvWS'),
        norm_cfg=dict(type='GN', num_groups=32))

    model.init_weights()
    model.train()

    input = _demo_inputs((1, 14, 320, 320))

    output = model(input)

    assert 'conv_out' in output.keys()
    assert 'merged' in output.keys()
    assert 'two_channel_trimap' in output.keys()

    assert isinstance(output['conv_out'], list)
    assert len(output['conv_out']) == 6

    assert isinstance(output['merged'], torch.Tensor)
    assert_tensor_with_shape(output['merged'], torch.Size([1, 3, 320, 320]))

    assert isinstance(output['two_channel_trimap'], torch.Tensor)
    assert_tensor_with_shape(output['two_channel_trimap'],
                             torch.Size([1, 2, 320, 320]))
    if torch.cuda.is_available():
        model = FBAResnetDilated(
            depth=50,
            in_channels=11,
            stem_channels=64,
            base_channels=64,
            conv_cfg=dict(type='ConvWS'),
            norm_cfg=dict(type='GN', num_groups=32))
        model.init_weights()
        model.train()
        model.cuda()

        input = _demo_inputs((1, 14, 320, 320)).cuda()
        output = model(input)

        assert 'conv_out' in output.keys()
        assert 'merged' in output.keys()
        assert 'two_channel_trimap' in output.keys()

        assert isinstance(output['conv_out'], list)
        assert len(output['conv_out']) == 6

        assert isinstance(output['merged'], torch.Tensor)
        assert_tensor_with_shape(output['merged'], torch.Size([1, 3, 320,
                                                               320]))

        assert isinstance(output['two_channel_trimap'], torch.Tensor)
        assert_tensor_with_shape(output['two_channel_trimap'],
                                 torch.Size([1, 2, 320, 320]))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
