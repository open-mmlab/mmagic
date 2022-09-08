# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.utils import collect_env


def test_collect_env():
    env_info = collect_env()
    assert isinstance(env_info, dict)
