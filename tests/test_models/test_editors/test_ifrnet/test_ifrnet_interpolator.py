# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models import IFRNetInterpolator


def test_ifrnet_interpolator():

    model = IFRNetInterpolator()

    # test attributes
    assert model.__class__.__name__ == 'IFRNetInterpolator'

    # prepare data
    img0 = torch.rand(1, 3, 64, 64)
    img1 = torch.rand(1, 3, 64, 64)
    embt = torch.rand(1, 1, 1, 1)

    gt = torch.rand(1, 3, 64, 64)
    gt_feats = [
        torch.rand(1, 32, 32, 32),
        torch.rand(1, 48, 16, 16),
        torch.rand(1, 72, 8, 8)
    ]
    gt_flow = [
        torch.rand(1, 2, 64, 64),
        torch.rand(1, 2, 32, 32),
        torch.rand(1, 2, 16, 16),
        torch.rand(1, 2, 8, 8)
    ]

    # test on cpu
    output = model(img0, img1, embt)
    assert isinstance(output, dict)
    print(output['pred_img'].shape)
    assert output['pred_img'].shape == gt.shape
    for i in range(3):
        assert gt_feats[i].shape == output['feats'][i].shape
    for i in range(4):
        assert gt_flow[i].shape == output['flows0'][i].shape == output[
            'flows1'][i].shape
