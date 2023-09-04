# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors import IndexNetDecoder, IndexNetEncoder


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
