# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter


def inpainting_inference(model, masked_img, mask):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        masked_img (str): File path of image with mask.
        mask (str): Mask file path.

    Returns:
        Tensor: The predicted inpainting result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    data = dict(gt_path=masked_img, mask_path=mask)
    _data = test_pipeline(data)
    data = dict()
    data['inputs'] = _data['inputs'] / 255.0
    data = collate([data])
    data['data_samples'] = [_data['data_samples']]
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
        data['data_samples'][0].mask.data = scatter(
            data['data_samples'][0].mask.data, [device])[0] / 255.0
    # else:
    #     data.pop('meta')
    # forward the model
    with torch.no_grad():
        result, x = model(mode='tensor', **data)

    masks = _data['data_samples'].mask.data * 255
    masked_imgs = data['inputs'][0]
    result = result[0] * masks + masked_imgs * (1. - masks)
    return result
