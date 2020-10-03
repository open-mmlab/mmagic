from functools import partial

import torch
from mmcv.parallel import collate, scatter

from mmedit.apis.matting_inference import init_model
from mmedit.datasets.pipelines import Compose


def _demo_mm_inputs(test_pipeline, device):
    merged_img_path = './tests/data/merged/GT05.jpg'
    trimap_path = './tests/data/trimap/GT05.jpg'
    data = dict(merged_path=merged_img_path, trimap_path=trimap_path)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    return data


def generate_inputs_and_wrap_model(config_path, checkpoint_path, input_config):
    """
    Construct matting model according to config_path and checkpoint_path.
    """

    model = init_model(
        config_path, checkpoint_path, device=torch.device('cuda'))

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
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
    data = _demo_mm_inputs(test_pipeline, device)

    meta = data['meta']
    merged = data['merged']
    trimap = data['trimap']

    model.forward = partial(model.forward, meta=meta, test_mode=True)

    return model, (merged, trimap)
