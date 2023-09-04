# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors import FLAVR, FLAVRNet


def test_flavr():

    model = FLAVR(
        generator=dict(
            type='FLAVRNet',
            num_input_frames=4,
            num_output_frames=2,
            mid_channels_list=[64, 32, 16, 8],
            encoder_layers_list=[1, 1, 1, 1],
            bias=False,
            norm_cfg=None,
            join_type='concat',
            up_mode='transpose'),
        pixel_loss=dict(type='L1Loss'),
        required_frames=4)
    assert isinstance(model, FLAVR)
    assert isinstance(model.generator, FLAVRNet)
    assert model.pixel_loss.__class__.__name__ == 'L1Loss'

    input_tensors = torch.rand((1, 9, 3, 16, 16))
    input_tensors = model.split_frames(input_tensors)
    assert input_tensors.shape == (6, 4, 3, 16, 16), input_tensors.shape

    output_tensors = torch.rand((6, 1, 3, 16, 16))
    result = model.merge_frames(input_tensors, output_tensors)
    assert len(result) == 15
    assert result[0].shape == (16, 16, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
