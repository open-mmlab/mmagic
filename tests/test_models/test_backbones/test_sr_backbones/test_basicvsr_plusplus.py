# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.backbones.sr_backbones.basicvsr_pp import BasicVSRPlusPlus


def test_basicvsr_plusplus():
    """Test BasicVSR++."""

    # cpu
    model = BasicVSRPlusPlus(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained=None,
        cpu_cache_length=100)
    input_tensor = torch.rand(1, 5, 3, 64, 64)
    model.init_weights(pretrained=None)
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    # with cpu_cache (no effect on cpu)
    model = BasicVSRPlusPlus(
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

    with pytest.raises(TypeError):
        # pretrained should be str or None
        model.init_weights(pretrained=[1])

    # output has the same size as input
    model = BasicVSRPlusPlus(
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
        model = BasicVSRPlusPlus(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
            spynet_pretrained=None,
            cpu_cache_length=100).cuda()
        input_tensor = torch.rand(1, 5, 3, 64, 64).cuda()
        model.init_weights(pretrained=None)
        output = model(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

        # with cpu_cache
        model = BasicVSRPlusPlus(
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

        with pytest.raises(TypeError):
            # pretrained should be str or None
            model.init_weights(pretrained=[1]).cuda()

        # output has the same size as input
        model = BasicVSRPlusPlus(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=False,
            spynet_pretrained=None,
            cpu_cache_length=100).cuda()
        input_tensor = torch.rand(1, 5, 3, 256, 256).cuda()
        output = model(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)
