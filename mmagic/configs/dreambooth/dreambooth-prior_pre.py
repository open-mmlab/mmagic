# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .dreambooth import *

# config for model
model.update(dict(prior_loss_weight=1, class_prior_prompt='a dog'))
