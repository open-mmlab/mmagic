# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.animatediff.motion_module import (
    FeedForward, VanillaTemporalModule)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
def test_VanillaTemporalModule():
    # This also test TemporalTransformerBlock
    input = torch.rand((1, 64, 16, 32, 32))
    text_feat = torch.rand([1, 77, 768])
    transformer = VanillaTemporalModule(in_channels=64)
    output = transformer.forward(input, 10, text_feat)
    assert output.shape == (1, 64, 16, 32, 32)


def test_FeedForward():
    input = torch.rand((2, 64, 64))
    feed_forward = FeedForward(64, 64, activation_fn='geglu')
    output = feed_forward.forward(input)
    assert output.shape == (2, 64, 64)


# if __name__ == '__main__':
#     test_VanillaTemporalModule()
#     test_FeedForward()
