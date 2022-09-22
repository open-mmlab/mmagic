# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.base_archs import ResidualBlockNoBN
from mmedit.models.utils import make_layer


def test_sr_backbone_utils():
    block = make_layer(ResidualBlockNoBN, 3)
    input = torch.rand((2, 64, 128, 128))
    output = block(input)
    assert output.detach().numpy().shape == (2, 64, 128, 128)
