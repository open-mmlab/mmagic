# Copyright (c) OpenMMLab. All rights reserved.
# TODO: remove this file
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('model', parent=MMCV_MODELS)
COMPONENTS = MODELS
LOSSES = MODELS
