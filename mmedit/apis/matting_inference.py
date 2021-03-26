import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmedit.datasets.pipelines import Compose
from mmedit.models import build_model


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


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
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['pred_alpha']
