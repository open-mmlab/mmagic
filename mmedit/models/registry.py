# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('model', parent=MMCV_MODELS)
BACKBONES = MODELS
COMPONENTS = MODELS
LOSSES = MODELS
