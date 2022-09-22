# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.stylegan3.stylegan3_modules import MappingNetwork


def test_MappingNetwork():
    mapping_network = MappingNetwork(16, 4, 5, c_dim=8)
    z = torch.randn(1, 16)
    c = torch.randn(1, 8)
    out = mapping_network(z, c)
    assert out.shape == (1, 5, 4)
