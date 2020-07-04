import torch.nn as nn
from mmcv import build_from_cfg

from .registry import BACKBONES, COMPONENTS, LOSSES, MODELS


def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone.

    Args:
        cfg (dict): Configuration for building backbone.
    """
    return build(cfg, BACKBONES)


def build_component(cfg):
    """Build component.

    Args:
        cfg (dict): Configuration for building component.
    """
    return build(cfg, COMPONENTS)


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (dict): Configuration for building loss.
    """
    return build(cfg, LOSSES)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
