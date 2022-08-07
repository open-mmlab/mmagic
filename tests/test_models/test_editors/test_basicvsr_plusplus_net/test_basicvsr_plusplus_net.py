# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.video_restorers import BasicVSRPlusPlusNet


def test_basicvsr_plusplus():
    """Test BasicVSR++."""

    # cpu
    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained=None,
        cpu_cache_length=100)
    input_tensor = torch.rand(1, 5, 3, 64, 64)
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    # with cpu_cache (no effect on cpu)
    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained=None,
        cpu_cache_length=3)
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    with pytest.raises(AssertionError):
        # The height and width of inputs should be at least 64
        input_tensor = torch.rand(1, 5, 3, 61, 61)
        model(input_tensor)

    # output has the same size as input
    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained=None,
        cpu_cache_length=100)
    input_tensor = torch.rand(1, 5, 3, 256, 256)
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    # gpu
    if torch.cuda.is_available():
        model = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
            spynet_pretrained=None,
            cpu_cache_length=100).cuda()
        input_tensor = torch.rand(1, 5, 3, 64, 64).cuda()
        output = model(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

        # with cpu_cache
        model = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
            spynet_pretrained=None,
            cpu_cache_length=3).cuda()
        output = model(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

        with pytest.raises(AssertionError):
            # The height and width of inputs should be at least 64
            input_tensor = torch.rand(1, 5, 3, 61, 61).cuda()
            model(input_tensor)

        # output has the same size as input
        model = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=False,
            spynet_pretrained=None,
            cpu_cache_length=100).cuda()
        input_tensor = torch.rand(1, 5, 3, 256, 256).cuda()
        output = model(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)
