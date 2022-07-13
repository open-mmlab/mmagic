# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.video_restorers.basicvsr import BasicVSRNet


def test_basicvsr_net():
    """Test BasicVSR."""

    # cpu
    basicvsr = BasicVSRNet(
        mid_channels=64, num_blocks=30, spynet_pretrained=None)
    input_tensor = torch.rand(1, 5, 3, 64, 64)
    output = basicvsr(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    # gpu
    if torch.cuda.is_available():
        basicvsr = BasicVSRNet(
            mid_channels=64, num_blocks=30, spynet_pretrained=None).cuda()
        input_tensor = torch.rand(1, 5, 3, 64, 64).cuda()
        output = basicvsr(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

    with pytest.raises(AssertionError):
        # The height and width of inputs should be at least 64
        input_tensor = torch.rand(1, 5, 3, 61, 61)
        basicvsr(input_tensor)
