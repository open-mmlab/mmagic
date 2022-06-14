# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.registry import (DATASETS, METRICS, MODELS, TRANSFORMS,
                             register_all_modules)


def test_register_all_modules():
    # but maybe already registered in other unittests
    register_all_modules()
    assert len(DATASETS)
    assert len(METRICS)
    assert len(MODELS)
    assert len(TRANSFORMS)
