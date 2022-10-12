# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter


def colorization_inference(model, img):

    device = next(model.parameters()).device

    # build the data pipeline
    test_pipeline = Compose(model.cfg.test_pipeline)
    # prepare data
    data = dict(img_path=img)
    data = test_pipeline(data)
    data = collate([data])

    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(mode='predict', **data)

    return result['fake_img']
