# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.models import IndexedUpsample, IndexNetDecoder, IndexNetEncoder


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def _demo_inputs(input_shape=(1, 4, 64, 64)):
    """
    Create a superset of inputs needed to run encoder.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 4, 64, 64).
    """
    img = np.random.random(input_shape).astype(np.float32)
    img = torch.from_numpy(img)

    return img


def test_indexed_upsample():
    """Test indexed upsample module for indexnet decoder."""
    indexed_upsample = IndexedUpsample(12, 12)

    # test indexed_upsample without dec_idx_feat (no upsample)
    x = torch.rand(2, 6, 32, 32)
    shortcut = torch.rand(2, 6, 32, 32)
    output = indexed_upsample(x, shortcut)
    assert_tensor_with_shape(output, (2, 12, 32, 32))

    # test indexed_upsample without dec_idx_feat (with upsample)
    x = torch.rand(2, 6, 32, 32)
    dec_idx_feat = torch.rand(2, 6, 64, 64)
    shortcut = torch.rand(2, 6, 64, 64)
    output = indexed_upsample(x, shortcut, dec_idx_feat)
    assert_tensor_with_shape(output, (2, 12, 64, 64))


def test_indexnet_decoder():
    """Test Indexnet decoder."""
    # test indexnet decoder with default indexnet setting
    with pytest.raises(AssertionError):
        # shortcut must have four dimensions
        indexnet_decoder = IndexNetDecoder(
            160, kernel_size=5, separable_conv=False)
        x = torch.rand(2, 256, 4, 4)
        shortcut = torch.rand(2, 128, 8, 8, 8)
        dec_idx_feat = torch.rand(2, 128, 8, 8, 8)
        outputs_enc = dict(
            out=x, shortcuts=[shortcut], dec_idx_feat_list=[dec_idx_feat])
        indexnet_decoder(outputs_enc)

    indexnet_decoder = IndexNetDecoder(
        160, kernel_size=5, separable_conv=False)
    indexnet_decoder.init_weights()
    indexnet_encoder = IndexNetEncoder(4)
    x = torch.rand(2, 4, 32, 32)
    outputs_enc = indexnet_encoder(x)
    out = indexnet_decoder(outputs_enc)
    assert out.shape == (2, 1, 32, 32)

    # test indexnet decoder with other setting
    indexnet_decoder = IndexNetDecoder(160, kernel_size=3, separable_conv=True)
    indexnet_decoder.init_weights()
    out = indexnet_decoder(outputs_enc)
    assert out.shape == (2, 1, 32, 32)
