from mmcv.cnn import build_model_from_cfg
from mmcv.utils import Registry

MODELS = Registry('models', build_func=build_model_from_cfg)

BACKBONES = MODELS
COMPONENTS = MODELS
LOSSES = MODELS


def build_backbone(cfg):
    """Build backbone.

    Args:
        cfg (dict): Configuration for building backbone.
    """
    return BACKBONES.build(cfg)


def build_component(cfg):
    """Build component.

    Args:
        cfg (dict): Configuration for building component.
    """
    return COMPONENTS.build(cfg)


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (dict): Configuration for building loss.
    """
    return LOSSES.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    cfg.update(dict(train_cfg=train_cfg, test_cfg=test_cfg))
    return MODELS.build(cfg)
