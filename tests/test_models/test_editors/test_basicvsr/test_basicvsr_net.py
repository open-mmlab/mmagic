# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors import BasicVSRNet


def test_basicvsr_net():
    """Test BasicVSR."""

    # cpu
    basicvsr = BasicVSRNet(
        mid_channels=8, num_blocks=1, spynet_pretrained=None)

    input_tensor = torch.rand(1, 5, 3, 64, 64)
    basicvsr(input_tensor)
    assert not basicvsr._raised_warning

    input_tensor = torch.rand(1, 5, 3, 16, 16)
    output = basicvsr(input_tensor)
    assert output.shape == (1, 5, 3, 64, 64)
    assert basicvsr._raised_warning

    # gpu
    if torch.cuda.is_available():
        basicvsr = BasicVSRNet(
            mid_channels=8, num_blocks=1, spynet_pretrained=None).cuda()
        input_tensor = torch.rand(1, 5, 3, 16, 16).cuda()
        output = basicvsr(input_tensor)
        assert output.shape == (1, 5, 3, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
