import torch.nn as nn
from mmcv.utils import build_from_cfg

from .registry import BACKBONES, COMPONENTS, LOSSES


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_component(cfg):
    return build(cfg, COMPONENTS)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_loss(cfg):
    return build(cfg, LOSSES)
