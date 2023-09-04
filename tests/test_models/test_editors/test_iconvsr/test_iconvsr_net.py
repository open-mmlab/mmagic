# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors import IconVSRNet


def test_iconvsr():
    """Test IconVSRNet."""

    iconvsr = IconVSRNet(
        mid_channels=64,
        num_blocks=1,
        keyframe_stride=5,
        padding=2,
        spynet_pretrained=None,
        edvr_pretrained=None)

    input_tensor = torch.rand(1, 5, 3, 64, 64)
    output = iconvsr(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)
    assert not iconvsr._raised_warning

    input_tensor = torch.rand(1, 5, 3, 16, 16)
    output = iconvsr(input_tensor)
    assert output.shape == (1, 5, 3, 64, 64)
    assert iconvsr._raised_warning

    # spynet_pretrained should be str or None
    with pytest.raises(TypeError):
        iconvsr = IconVSRNet(
            mid_channels=64,
            num_blocks=1,
            keyframe_stride=5,
            padding=2,
            spynet_pretrained=123,
            edvr_pretrained=None)

    # edvr_pretrained should be str or None
    with pytest.raises(TypeError):
        iconvsr = IconVSRNet(
            mid_channels=64,
            num_blocks=1,
            keyframe_stride=5,
            padding=2,
            spynet_pretrained=None,
            edvr_pretrained=123)

    if torch.cuda.is_available():
        iconvsr = IconVSRNet(
            mid_channels=64,
            num_blocks=1,
            keyframe_stride=5,
            padding=2,
            spynet_pretrained=None,
            edvr_pretrained=None).cuda()

        # input_tensor = torch.rand(1, 5, 3, 64, 64).cuda()
        input_tensor = torch.rand(1, 5, 3, 16, 16).cuda()
        output = iconvsr(input_tensor)
        # assert output.shape == (1, 5, 3, 256, 256)
        assert output.shape == (1, 5, 3, 64, 64)

        # spynet_pretrained should be str or None
        with pytest.raises(TypeError):
            iconvsr = IconVSRNet(
                mid_channels=64,
                num_blocks=1,
                keyframe_stride=5,
                padding=2,
                spynet_pretrained=123,
                edvr_pretrained=None).cuda()

        # edvr_pretrained should be str or None
        with pytest.raises(TypeError):
            iconvsr = IconVSRNet(
                mid_channels=64,
                num_blocks=1,
                keyframe_stride=5,
                padding=2,
                spynet_pretrained=None,
                edvr_pretrained=123).cuda()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
