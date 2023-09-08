# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.pix2pix.pix2pix_modules import \
    UnetSkipConnectionBlock


def test_unet_skip_connection_block():
    block = UnetSkipConnectionBlock(16, 16, is_innermost=True)
    input = torch.rand((2, 16, 128, 128))
    output = block(input)
    assert output.detach().numpy().shape == (2, 32, 128, 128)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
