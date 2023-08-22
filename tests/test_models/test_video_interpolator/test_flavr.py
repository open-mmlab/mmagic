# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.video_interpolators.flavr import FLAVR


def test_flavr():

    model = FLAVR(
        generator=dict(
            type='FLAVRNet', num_input_frames=4, num_output_frames=1),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

    input_tensors = torch.rand(3, 4, 3, 16, 16)
    output_tensors = torch.rand(3, 1, 3, 16, 16)
    result = model.merge_frames(input_tensors, output_tensors)
    assert len(result) == 9
    assert result[0].shape == (16, 16, 3)
