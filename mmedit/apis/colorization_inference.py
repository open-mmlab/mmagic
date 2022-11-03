# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter


def colorization_inference(model, img):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): Image file path.

    Returns:
        Tensor: The predicted colorization result.
    """
    device = next(model.parameters()).device

    # build the data pipeline
    test_pipeline = Compose(model.cfg.test_pipeline)
    # prepare data
    data = dict(img_path=img)
    _data = test_pipeline(data)
    data = dict()
    data['inputs'] = _data['inputs'] / 255.0
    data = collate([data])
    data['data_samples'] = [_data['data_samples']]
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
        if not data['data_samples'][0].empty_box:
            data['data_samples'][0].cropped_img.data = scatter(
                data['data_samples'][0].cropped_img.data, [device])[0] / 255.0

            data['data_samples'][0].box_info.data = scatter(
                data['data_samples'][0].box_info.data, [device])[0]

            data['data_samples'][0].box_info_2x.data = scatter(
                data['data_samples'][0].box_info_2x.data, [device])[0]

            data['data_samples'][0].box_info_4x.data = scatter(
                data['data_samples'][0].box_info_4x.data, [device])[0]

            data['data_samples'][0].box_info_8x.data = scatter(
                data['data_samples'][0].box_info_8x.data, [device])[0]

    # forward the model
    with torch.no_grad():
        result = model(mode='tensor', **data)

    return result
