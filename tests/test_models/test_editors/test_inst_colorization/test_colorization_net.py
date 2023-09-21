# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.registry import MODELS


def test_colorization_net():

    model_cfg = dict(
        type='ColorizationNet', input_nc=4, output_nc=2, norm_type='batch')

    # build model
    model = MODELS.build(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'ColorizationNet'

    # prepare data
    input_A = torch.rand(1, 1, 256, 256)
    input_B = torch.rand(1, 2, 256, 256)
    mask_B = torch.rand(1, 1, 256, 256)

    target_shape = (1, 2, 256, 256)

    # test on cpu
    (out_class, out_reg, feature_map) = model(input_A, input_B, mask_B)
    assert isinstance(feature_map, dict)
    assert feature_map['conv1_2'].shape == (1, 64, 256, 256) \
           and feature_map['out_reg'].shape == target_shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        input_A = input_A.cuda()
        input_B = input_B.cuda()
        mask_B = mask_B.cuda()
        (out_class, out_reg, feature_map) = \
            model(input_A, input_B, mask_B)

        assert isinstance(feature_map, dict)
        for item in feature_map.keys():
            assert torch.is_tensor(feature_map[item])


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
