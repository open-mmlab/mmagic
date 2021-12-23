# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.backbones import TOFlow


def test_tof():
    """Test TOFlow."""

    # cpu
    tof = TOFlow(adapt_official_weights=True)
    input_tensor = torch.rand(1, 7, 3, 32, 32)
    tof.init_weights(pretrained=None)
    output = tof(input_tensor)
    assert output.shape == (1, 3, 32, 32)

    tof = TOFlow(adapt_official_weights=False)
    tof.init_weights(pretrained=None)
    output = tof(input_tensor)
    assert output.shape == (1, 3, 32, 32)

    with pytest.raises(TypeError):
        # pretrained should be str or None
        tof.init_weights(pretrained=[1])

    # gpu
    if torch.cuda.is_available():
        tof = TOFlow(adapt_official_weights=True).cuda()
        input_tensor = torch.rand(1, 7, 3, 32, 32).cuda()
        tof.init_weights(pretrained=None)
        output = tof(input_tensor)
        assert output.shape == (1, 3, 32, 32)
