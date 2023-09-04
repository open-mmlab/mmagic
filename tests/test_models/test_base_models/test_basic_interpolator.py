# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from mmagic.models import BasicInterpolator
from mmagic.models.losses import L1Loss
from mmagic.registry import MODELS


@MODELS.register_module()
class InterpolateExample(nn.Module):
    """An example of interpolate network for testing BasicInterpolator."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        return self.layer(x[:, 0])

    def init_weights(self, pretrained=None):
        pass


def test_basic_interpolator():

    model = BasicInterpolator(
        generator=dict(type='InterpolateExample'),
        pixel_loss=dict(type='L1Loss'))
    assert model.__class__.__name__ == 'BasicInterpolator'
    assert isinstance(model.generator, InterpolateExample)
    assert isinstance(model.pixel_loss, L1Loss)

    input_tensors = torch.rand((1, 9, 3, 16, 16))
    input_tensors = model.split_frames(input_tensors)
    assert input_tensors.shape == (8, 2, 3, 16, 16)

    output_tensors = torch.rand((8, 1, 3, 16, 16))
    result = model.merge_frames(input_tensors, output_tensors)
    assert len(result) == 17
    assert result[0].shape == (16, 16, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
