# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter


def matting_inference(model, img, trimap):
    """Inference image(s) with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): Image file path.
        trimap (str): Trimap file path.

    Returns:
        np.ndarray: The predicted alpha matte.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove alpha from test_pipeline
    keys_to_remove = ['alpha', 'ori_alpha']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    data = dict(merged_path=img, trimap_path=trimap)
    _data = test_pipeline(data)
    trimap = _data['data_samples'].trimap.data
    data = dict()
    data['inputs'] = torch.cat([_data['inputs'], trimap], dim=0).float()
    data = collate([data])
    data['data_samples'] = [_data['data_samples']]
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(mode='predict', **data)
    result = result[0].output
    result = result.pred_alpha.data
    return result.cpu().numpy()
