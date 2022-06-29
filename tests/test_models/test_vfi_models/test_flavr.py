# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.video_interpolators import FLAVRNet
from mmedit.models.video_interpolators.flavr.flavr import FLAVR
from mmedit.models.video_interpolators.flavr.flavr_net import UpConv3d
from mmedit.registry import MODELS


def test_flavr_net():

    model_cfg = dict(
        type='FLAVRNet',
        num_input_frames=4,
        num_output_frames=2,
        mid_channels_list=[64, 32, 16, 8],
        encoder_layers_list=[1, 1, 1, 1],
        bias=False,
        norm_cfg=None,
        join_type='concat',
        up_mode='transpose')

    # build model
    model = MODELS.build(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'FLAVRNet'

    # prepare data
    inputs = torch.rand(1, 4, 3, 256, 256)
    target = torch.rand(1, 2, 3, 256, 256)

    # test on cpu
    output = model(inputs)
    output = model(inputs)
    assert torch.is_tensor(output)
    assert output.shape == target.shape

    out = model.decoder._join_tensors(inputs, inputs)
    assert out.shape == (1, 8, 3, 256, 256)

    conv3d = UpConv3d(4, 4, 1, 1, 1, up_mode='trilinear', batchnorm=True)
    out = conv3d(inputs)
    assert out.shape == (1, 4, 3, 512, 512)

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        output = model(inputs)
        output = model(inputs)
        assert torch.is_tensor(output)
        assert output.shape == target.shape


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
