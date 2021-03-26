import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose


def restoration_inference(model, img):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        Tensor: The predicted restoration result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove gt from test_pipeline
    keys_to_remove = ['gt']
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
    data = dict(lq_path=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['output']
