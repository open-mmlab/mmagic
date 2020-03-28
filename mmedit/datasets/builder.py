from mmcv.utils import build_from_cfg

from .registry import DATASETS


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset
