# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmagic.registry import MODELS


def test_fusion_net():

    model_cfg = dict(
        type='FusionNet', input_nc=4, output_nc=2, norm_type='batch')

    # build model
    model = MODELS.build(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'FusionNet'

    # prepare data
    input_A = torch.rand(1, 1, 256, 256)
    input_B = torch.rand(1, 2, 256, 256)
    mask_B = torch.rand(1, 1, 256, 256)

    instance_feature = dict(
        conv1_2=torch.rand(1, 64, 256, 256),
        conv2_2=torch.rand(1, 128, 128, 128),
        conv3_3=torch.rand(1, 256, 64, 64),
        conv4_3=torch.rand(1, 512, 32, 32),
        conv5_3=torch.rand(1, 512, 32, 32),
        conv6_3=torch.rand(1, 512, 32, 32),
        conv7_3=torch.rand(1, 512, 32, 32),
        conv8_up=torch.rand(1, 256, 64, 64),
        conv8_3=torch.rand(1, 256, 64, 64),
        conv9_up=torch.rand(1, 128, 128, 128),
        conv9_3=torch.rand(1, 128, 128, 128),
        conv10_up=torch.rand(1, 128, 256, 256),
        conv10_2=torch.rand(1, 128, 256, 256),
    )

    target_shape = (1, 2, 256, 256)

    box_info_box = [
        torch.tensor([[175, 29, 96, 54, 52, 106], [14, 191, 84, 61, 51, 111],
                      [117, 64, 115, 46, 75, 95], [41, 165, 121, 47, 50, 88],
                      [46, 136, 94, 45, 74, 117], [79, 124, 62, 115, 53, 79],
                      [156, 64, 77, 138, 36, 41], [200, 48, 114, 131, 8, 11],
                      [115, 78, 92, 81, 63, 83]]),
        torch.tensor([[87, 15, 48, 27, 26, 53], [7, 96, 42, 31, 25, 55],
                      [58, 32, 57, 23, 38, 48], [20, 83, 60, 24, 25, 44],
                      [23, 68, 47, 23, 37, 58], [39, 62, 31, 58, 27, 39],
                      [78, 32, 38, 69, 18, 21], [100, 24, 57, 66, 4, 5],
                      [57, 39, 46, 41, 32, 41]]),
        torch.tensor([[43, 8, 24, 14, 13, 26], [3, 48, 21, 16, 13, 27],
                      [29, 16, 28, 12, 19, 24], [10, 42, 30, 12, 12, 22],
                      [11, 34, 23, 12, 19, 29], [19, 31, 15, 29, 14, 20],
                      [39, 16, 19, 35, 9, 10], [50, 12, 28, 33, 2, 3],
                      [28, 20, 23, 21, 16, 20]]),
        torch.tensor([[21, 4, 12, 7, 7, 13], [1, 24, 10, 8, 7, 14],
                      [14, 8, 14, 6, 10, 12], [5, 21, 15, 6, 6, 11],
                      [5, 17, 11, 6, 10, 15], [9, 16, 7, 15, 7, 10],
                      [19, 8, 9, 18, 5, 5], [25, 6, 14, 17, 1, 1],
                      [14, 10, 11, 11, 8, 10]])
    ]

    # test on cpu
    out = model(input_A, input_B, mask_B, instance_feature, box_info_box)
    assert torch.is_tensor(out)
    assert out.shape == target_shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        input_A = input_A.cuda()
        input_B = input_B.cuda()
        mask_B = mask_B.cuda()
        for item in instance_feature.keys():
            instance_feature[item] = instance_feature[item].cuda()
        box_info_box = [i.cuda() for i in box_info_box]
        output = model(input_A, input_B, mask_B, instance_feature,
                       box_info_box)
        assert torch.is_tensor(output)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
