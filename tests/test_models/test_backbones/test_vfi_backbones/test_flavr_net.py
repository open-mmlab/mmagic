# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models import build_backbone
from mmedit.models.backbones.vfi_backbones.flavr_net import UpConv3d


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
    model = build_backbone(model_cfg)

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
    with pytest.raises(OSError):
        model.init_weights('')
    with pytest.raises(TypeError):
        model.init_weights(1)

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
