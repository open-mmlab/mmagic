# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.backbones.sr_backbones.iconvsr import IconVSR


def test_iconvsr():
    """Test IconVSR."""

    # gpu (since IconVSR contains DCN, only GPU mode is available)
    if torch.cuda.is_available():
        iconvsr = IconVSR(
            mid_channels=64,
            num_blocks=30,
            keyframe_stride=5,
            padding=2,
            spynet_pretrained=None,
            edvr_pretrained=None).cuda()

        input_tensor = torch.rand(1, 5, 3, 64, 64).cuda()
        iconvsr.init_weights(pretrained=None)
        output = iconvsr(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

        with pytest.raises(AssertionError):
            # The height and width of inputs should be at least 64
            input_tensor = torch.rand(1, 5, 3, 61, 61)
            iconvsr(input_tensor)

        with pytest.raises(TypeError):
            # pretrained should be str or None
            iconvsr.init_weights(pretrained=[1])

        # spynet_pretrained should be str or None
        with pytest.raises(TypeError):
            iconvsr = IconVSR(
                mid_channels=64,
                num_blocks=30,
                keyframe_stride=5,
                padding=2,
                spynet_pretrained=123,
                edvr_pretrained=None).cuda()

        # edvr_pretrained should be str or None
        with pytest.raises(TypeError):
            iconvsr = IconVSR(
                mid_channels=64,
                num_blocks=30,
                keyframe_stride=5,
                padding=2,
                spynet_pretrained=None,
                edvr_pretrained=123).cuda()
