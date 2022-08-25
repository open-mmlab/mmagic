# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose


def colorization_inference(model, img):
    device = next(model.parameters()).device

    test_pipeline = [
        dict(type='LoadImageFromFile', key='gt'),
        dict(type='GenMaskRCNNBbox', stage='test_fusion'),
        dict(
            type='Collect',
            keys=[
                'full_img', 'empty_box', 'cropped_img', 'box_info',
                'box_info_2x', 'box_info_4x', 'box_info_8x'
            ],
            meta_keys=[])
    ]
    # build the data pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(gt_path=img)
    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)

    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
    else:
        data.pop('meta')
    # print(data.size)
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['fake_img']
