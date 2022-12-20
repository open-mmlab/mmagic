# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

METRICS = Registry('metric')


def build_metric(cfg):
    """Build a metric calculator."""
    return build_from_cfg(cfg, METRICS)
