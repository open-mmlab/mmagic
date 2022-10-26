# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models.base_models import BaseBackbone


def test_base_backbone():
    base_backbone = BaseBackbone()
    base_backbone.init_weights()
