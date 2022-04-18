# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose


def inpainting_inference(model, masked_img, mask):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        masked_img (str): File path of image with mask.
        mask (str): Mask file path.

    Returns:
        Tensor: The predicted inpainting result.
    """
    device = next(model.parameters()).device  # model device

    infer_pipeline = [
        dict(type='LoadImageFromFile', key='masked_img'),
        dict(type='LoadMask', mask_mode='file', mask_config=dict()),
        dict(type='Pad', keys=['masked_img', 'mask'], mode='reflect'),
        dict(
            type='Normalize',
            keys=['masked_img'],
            mean=[127.5] * 3,
            std=[127.5] * 3,
            to_rgb=False),
        dict(type='GetMaskedImage', img_name='masked_img'),
        dict(
            type='Collect',
            keys=['masked_img', 'mask'],
            meta_keys=['masked_img_path']),
        dict(type='ImageToTensor', keys=['masked_img', 'mask'])
    ]

    # build the data pipeline
    test_pipeline = Compose(infer_pipeline)
    # prepare data
    data = dict(masked_img_path=masked_img, mask_path=mask)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
    else:
        data.pop('meta')
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['fake_img']
