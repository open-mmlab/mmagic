# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import collate, scatter
from mmengine.dataset import Compose


def restoration_inference(model, img, ref=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
        ref (str | None): File path of reference image. Default: None.

    Returns:
        Tensor: The predicted restoration result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
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
    if ref:  # Ref-SR
        data = dict(img_path=img, ref_path=ref)
    else:  # SISR
        data = dict(img_path=img)
    _data = test_pipeline(data)
    data = dict()
    data['batch_inputs'] = _data['inputs'] / 255.0
    data = collate([data], samples_per_gpu=1)
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(mode='tensor', **data)
    result = torch.stack([result[0][2], result[0][1], result[0][0]], dim=0)
    return result
