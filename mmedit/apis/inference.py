# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Config
from mmengine.config import ConfigDict
from mmengine.runner import load_checkpoint
from mmengine.runner import set_random_seed as set_random_seed_engine

from mmedit.registry import MODELS
from mmedit.utils import register_all_modules


def set_random_seed(seed, deterministic=False, use_rank_shift=True):
    """Set random seed.

    In this function, we just modify the default behavior of the similar
    function defined in MMCV.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: True.
    """
    set_random_seed_engine(
        seed, deterministic=deterministic, use_rank_shift=use_rank_shift)


def delete_cfg(cfg, key='init_cfg'):
    """Delete key from config object.

    Args:
        cfg (str or :obj:`mmengine.Config`): Config object.
        key (str): Which key to delete.
    """

    if key in cfg:
        cfg.pop(key)
    for _key in cfg.keys():
        if isinstance(cfg[_key], ConfigDict):
            delete_cfg(cfg[_key], key)


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # config.test_cfg.metrics = None
    delete_cfg(config.model, 'init_cfg')

    register_all_modules()
    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model
