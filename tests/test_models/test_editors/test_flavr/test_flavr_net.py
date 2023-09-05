# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.flavr.flavr_net import UpConv3d
from mmagic.registry import MODELS


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
