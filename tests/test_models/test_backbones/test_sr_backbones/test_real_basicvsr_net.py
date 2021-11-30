# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.backbones.sr_backbones.real_basicvsr_net import \
    RealBasicVSRNet


def test_real_basicvsr_net():
    """Test RealBasicVSR."""

    # cpu
    real_basicvsr = RealBasicVSRNet(is_fix_cleaning=False)
    real_basicvsr = RealBasicVSRNet(is_fix_cleaning=True)
    input_tensor = torch.rand(1, 5, 3, 64, 64)
    real_basicvsr.init_weights(pretrained=None)
    output = real_basicvsr(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    # gpu
    if torch.cuda.is_available():
        real_basicvsr = RealBasicVSRNet(is_fix_cleaning=False).cuda()
        real_basicvsr = RealBasicVSRNet(is_fix_cleaning=True).cuda()
        input_tensor = torch.rand(1, 5, 3, 64, 64).cuda()
        real_basicvsr.init_weights(pretrained=None)
        output = real_basicvsr(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

    with pytest.raises(TypeError):
        # pretrained should be str or None
        real_basicvsr.init_weights(pretrained=[1])
