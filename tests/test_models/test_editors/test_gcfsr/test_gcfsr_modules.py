# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
import torch

from mmedit.models.editors.gcfsr import GCFSR, GCFSR_blind


def _dummy_inputs(input_shape=(1, 3, 128, 128)):
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
    assert torch.isnan(tensor).sum() == 0
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def test_gcfsr():
    """Test GCFSR and GCFSR_blind on CPU device."""

    gcfsr = GCFSR(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        resample_kernel=(1, 3, 3, 1),
        narrow=1,
    )
    gcfsr_blind = GCFSR_blind(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        resample_kernel=(1, 3, 3, 1),
        narrow=1,
    )

    scale = random.choice([4, 8, 16, 32, 64])
    shape = (1, 3, 512, 512)
    img = _dummy_inputs(input_shape=shape)
    in_size = scale / 64.
    cond = torch.from_numpy(np.array([in_size], dtype=np.float32)).unsqueeze(0)
    output, _ = gcfsr(img, in_size=cond)
    assert_tensor_with_shape(output, shape)
    del gcfsr

    output_blind, _ = gcfsr_blind(img)
    assert_tensor_with_shape(output_blind, shape)
    del gcfsr_blind
