# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.animatediff.attention_3d import (CrossAttention,
                                                            Transformer3DModel)


def test_crossattention():
    input = torch.rand((2, 64, 64))
    crossattention = CrossAttention(64)
    crossattention.set_attention_slice(2)
    output = crossattention.forward(input)
    assert output.shape == (2, 64, 64)
    with pytest.raises(Exception):
        crossattention.set_attention_slice(10)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
def test_Transformer3DModel_init():
    with pytest.raises(Exception):
        Transformer3DModel(
            in_channels=32,
            num_vector_embeds=4,
            unet_use_cross_frame_attention=False,
            unet_use_temporal_attention=False)

    with pytest.raises(Exception):
        Transformer3DModel()

    Transformer3DModel(
        in_channels=32,
        use_linear_projection=True,
        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False)


# if __name__ == '__main__':
#     test_crossattention()
#     test_Transformer3DModel_init()
