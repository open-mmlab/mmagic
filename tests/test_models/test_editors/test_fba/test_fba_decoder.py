# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmagic.models.editors import FBADecoder


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def _demo_inputs(input_shape=(1, 4, 64, 64)):
    """Create a superset of inputs needed to run encoder.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 4, 64, 64).
    """
    img = np.random.random(input_shape).astype(np.float32)
    img = torch.from_numpy(img)

    return img


def test_fba_decoder():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        FBADecoder(pool_scales=1, in_channels=32, channels=16)
    inputs = dict()
    conv_out_1 = _demo_inputs((1, 11, 320, 320))
    conv_out_2 = _demo_inputs((1, 64, 160, 160))
    conv_out_3 = _demo_inputs((1, 256, 80, 80))
    conv_out_4 = _demo_inputs((1, 512, 40, 40))
    conv_out_5 = _demo_inputs((1, 1024, 40, 40))
    conv_out_6 = _demo_inputs((1, 2048, 40, 40))
    inputs['conv_out'] = [
        conv_out_1, conv_out_2, conv_out_3, conv_out_4, conv_out_5, conv_out_6
    ]
    inputs['merged'] = _demo_inputs((1, 3, 320, 320))
    inputs['two_channel_trimap'] = _demo_inputs((1, 2, 320, 320))
    model = FBADecoder(
        pool_scales=(1, 2, 3, 6),
        in_channels=2048,
        channels=256,
        norm_cfg=dict(type='GN', num_groups=32))

    alpha, F, B = model(inputs)
    assert_tensor_with_shape(alpha, torch.Size([1, 1, 320, 320]))
    assert_tensor_with_shape(F, torch.Size([1, 3, 320, 320]))
    assert_tensor_with_shape(B, torch.Size([1, 3, 320, 320]))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
